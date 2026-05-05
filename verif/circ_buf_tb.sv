`timescale 1ns/1ps

module circ_buf_tb;

    // ─── Parameters ────────────────────────────────────────────────────────────
    localparam WIDTH      = 8;
    localparam DEPTH      = 10;
    localparam CLK_PERIOD = 10; // ns

    // ─── DUT Signals ───────────────────────────────────────────────────────────
    logic                     clk;
    logic                     reset;
    logic [$clog2(DEPTH)-1:0] read_addr;
    logic [$clog2(DEPTH)-1:0] wr_ptr;
    logic                     wr_en;
    logic [WIDTH-1:0]         din;
    logic [WIDTH-1:0]         dout;

    integer idx;

    // ─── DUT Instantiation ─────────────────────────────────────────────────────
    circ_buf #(
        .WIDTH (WIDTH),
        .DEPTH (DEPTH)
    ) dut (
        .clk       (clk),
        .rst_n     (reset),      // fixed: was rst_n, module port is reset
        .read_addr (read_addr),
        .wr_en     (wr_en),
        .din       (din),
        .dout      (dout),
        .o_wr_ptr  (wr_ptr)
    );

    // ─── Clock Generation ──────────────────────────────────────────────────────
    initial clk = 0;
    always #(CLK_PERIOD / 2) clk = ~clk;

    // ─── Shadow memory & write pointer ─────────────────────────────────────────
    logic [WIDTH-1:0] expected_mem [0:DEPTH-1];
    int unsigned      write_count;

    // ─── Task : apply_reset ────────────────────────────────────────────────────
    // reset is active-LOW in the module (posedge clk or negedge rst_n),
    // so we pulse it low for one cycle then release high.
    task automatic apply_reset();
        $display("[%0t] >> RESET asserted", $time);
        @(negedge clk);          // change on negedge to avoid setup violations
        reset       = 0;         // assert (active-low)
        wr_en       = 0;
        din         = '0;
        read_addr   = '0;
        @(negedge clk);
        reset       = 1;         // release
        write_count = 0;
        @(posedge clk);      // let FSM settle into READ state
        @(posedge clk); #1;
        $display("[%0t] >> RESET released", $time);
    endtask

    // ─── Task : clear_memory ───────────────────────────────────────────────────
    // Each write takes TWO cycles through the FSM:
    //   cycle 1 (READ state)  : FSM sees wr_en, advances to WRITE
    //   cycle 2 (WRITE state) : ram_ptr <- wr_ptr, RAM write happens
    // We therefore wait two posedges per word before moving to the next.
    task automatic clear_memory();
        int i;
        $display("[%0t] >> CLEAR MEMORY — writing 0x00 to all %0d locations", $time, DEPTH);
        for (i = 0; i < DEPTH; i++) begin
            wr_en                             = 1;
            din                               = 8'h00;
            expected_mem[write_count % DEPTH] = 8'h00;
            write_count++;
            @(posedge clk); #1;   // READ  → WRITE transition
            @(posedge clk); #1;   // WRITE → READ  transition (RAM write committed)
            wr_en = 0;
        end
        $display("[%0t] >> CLEAR MEMORY done", $time);
    endtask

    // ─── Task : sequential_write ───────────────────────────────────────────────
    task automatic sequential_write(input logic [WIDTH-1:0] base_val);
        int i;
        logic [WIDTH-1:0] wdata;
        $display("[%0t] >> SEQUENTIAL WRITE — %0d writes starting from base 0x%02h",
                 $time, DEPTH, base_val);
        @(posedge clk); #1;
        for (i = 0; i < DEPTH; i++) begin
            wdata  = base_val + WIDTH'(i);
            wr_en  = 1;
            din    = wdata;
            expected_mem[write_count % DEPTH] = wdata;
            write_count++;
            @(posedge clk); #1;  // READ  → WRITE  (transition)
            @(posedge clk); #1;   // WRITE → READ   (RAM committed)
            $display("  [%0t]   wrote [addr=%0d] <= 0x%02h",
             $time, (write_count-1) % DEPTH, wdata);
end
        $display("[%0t] >> SEQUENTIAL WRITE done", $time);
    endtask

    // ─── Task : read_all_memory ────────────────────────────────────────────────
    // RAM is fully synchronous: present address, wait one posedge, dout is valid
    // the cycle after that posedge.
    // The FSM drives ram_ptr <- read_addr every READ cycle, so we stay out of
    // WRITE (wr_en=0) and simply clock through each address.
    task automatic read_all_memory(input string tag);
        int addr;
        logic [WIDTH-1:0] got;
        int errors;
        errors = 0;
        $display("[%0t] >> READ ALL (%s) — reading %0d locations", $time, tag, DEPTH);
        wr_en = 0;
        for (addr = 0; addr < DEPTH; addr++) begin
            read_addr = $clog2(DEPTH)'(addr);  // drive address
            @(posedge clk); #1;                // FSM latches ram_ptr = read_addr
            @(posedge clk); #1;                // RAM registers address, dout valid
            got = dout;
            if (got !== expected_mem[addr]) begin
                $display("  [%0t]   MISMATCH addr=%0d : got=0x%02h  expected=0x%02h",
                         $time, addr, got, expected_mem[addr]);
                errors++;
            end else begin
                $display("  [%0t]   OK      addr=%0d : 0x%02h", $time, addr, got);
            end
        end
        if (errors == 0)
            $display("[%0t] >> READ ALL (%s) PASSED", $time, tag);
        else
            $display("[%0t] >> READ ALL (%s) FAILED — %0d mismatch(es)", $time, tag, errors);
    endtask

    // ─── Task : burst_write ────────────────────────────────────────────────────
    task automatic burst_write(input int unsigned n, input logic [WIDTH-1:0] base_val);
        int i;
        logic [WIDTH-1:0] wdata;
        $display("[%0t] >> BURST WRITE — %0d writes from base 0x%02h", $time, n, base_val);
        for (i = 0; i < n; i++) begin
            wdata  = base_val + WIDTH'(i);
            wr_en  = 1;
            din    = wdata;
            expected_mem[write_count % DEPTH] = wdata;
            write_count++;
            @(posedge clk); #1;   // READ  → WRITE
            @(posedge clk); #1;   // WRITE → READ  (committed)
            wr_en = 0;
            $display("  [%0t]   wrote [addr=%0d] <= 0x%02h",
                     $time, (write_count-1) % DEPTH, wdata);
        end
        $display("[%0t] >> BURST WRITE done", $time);
    endtask

    // ─── Stimulus ──────────────────────────────────────────────────────────────
    initial begin
        reset       = 1;
        wr_en       = 0;
        din         = '0;
        read_addr   = '0;
        write_count = 0;

        apply_reset();

        clear_memory();
        read_all_memory("after clear");

        sequential_write(8'hA0);
        read_all_memory("after sequential write");

        burst_write(5, 8'h55);
        read_all_memory("after burst-5 write");

        $display("\n[%0t] *** SIMULATION COMPLETE ***\n", $time);
        $finish;
    end

    // ─── Timeout watchdog ──────────────────────────────────────────────────────
    initial begin
        #(CLK_PERIOD * 2000);
        $display("[%0t] WATCHDOG TIMEOUT — simulation halted", $time);
        $finish;
    end

    // ─── Waveform dump ─────────────────────────────────────────────────────────
    initial begin
        $dumpfile(`VCD_FILE);
        for (idx = 0; idx < DEPTH; idx = idx + 1) begin
            $dumpvars(0, dut.uut.ram_block[idx]);
            $dumpvars(0, expected_mem[idx]);
        end
        $dumpvars(0);
    end

endmodule