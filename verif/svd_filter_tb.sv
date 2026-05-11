`timescale 1ns/1ps

module svd_filter_tb;

    // ─────────────────────────────────────────────
    // Parameters
    // ─────────────────────────────────────────────
    parameter WIDTH = 8;
    parameter B     = 2;
    parameter C     = 81;
    parameter R     = 80;

    // ─────────────────────────────────────────────
    // DUT Signals
    // ─────────────────────────────────────────────
    reg                  clk;
    reg                  reset;
    reg                  x_clk;
    reg x_clk_en;
    reg  signed [WIDTH-1:0]     x;
    wire signed [WIDTH:0]     y;

    integer idx;

    // ─────────────────────────────────────────────
    // Clock Generation
    // clk   : 10ns period (50 MHz)
    // x_clk :  period (400 Hz) — Much Slower
    // ─────────────────────────────────────────────
    initial clk = 0;
    always #5  clk   = ~clk; 

    initial x_clk = 0;
    initial x_clk_en = 0;
    always #5000 x_clk = (x_clk_en) ? ~x_clk : 0;

    // ─────────────────────────────────────────────
    // DUT Instantiation
    // ─────────────────────────────────────────────
    svd_filter #(
        .WIDTH (WIDTH),
        .B     (B),
        .C     (C),
        .R     (R)
    ) dut (
        .clk   (clk),
        .reset (reset),
        .x_clk (x_clk),
        .x     (x),
        .y     (y),
        .y_done(y_done)
    );

    // ─────────────────────────────────────────────
    // Tasks
    // ─────────────────────────────────────────────

    // Apply reset
    task apply_reset;
        begin
            reset = 0;          // active low
            x     = 0;
            sample_idx = 0;
            ignore_first_sample = 0;
            repeat(4) @(posedge clk);
            reset = 1;
            @(posedge clk);
            $display("[%0t] Reset released", $time);
        end
    endtask

    // Send one sample on x_clk
    task send_sample;
        input [WIDTH-1:0] sample;
        begin
            @(posedge x_clk);
            x = sample;
            $display("[%0t] Input x = %0d (0x%02h)", $time, sample, sample);
        end
    endtask

    // Wait N system clock cycles
    task wait_cycles;
        input integer n;
        begin
            repeat(n) @(posedge clk);
        end
    endtask

    // ─────────────────────────────────────────────
    // Output Monitor
    // ─────────────────────────────────────────────
    /*always @(posedge clk) begin
        if (reset)
            $display("[%0t]   --> y = %0d (0x%02h)", $time, y, y);
    end
    */
    // ─────────────────────────────────────────────
    // Test Sequence
    // ─────────────────────────────────────────────
    integer i;
    parameter N_SAMPLES = 1000;


integer fd;
integer sample_idx;
logic ignore_first_sample;
    initial begin
        fd = $fopen("results.txt", "w");
        if (fd == 0) begin
            $display("ERROR: could not open results.txt");
            $finish;
        end
        $fdisplay(fd, "idx, x, y");  // CSV header
    end

    always @(posedge y_done) begin
        if (!ignore_first_sample) ignore_first_sample = 1;
        else begin $display("Output y = %d  (x = %d)", y, sample_mem[sample_idx - 1]);
            $fdisplay(fd, "%0d, %0d, %0d", sample_idx - 1, sample_mem[sample_idx - 1], y);
        end
        sample_idx <= sample_idx + 1;
    end

    // Close file cleanly at end of simulation
    final begin
        $fclose(fd);
    end

    logic [WIDTH-1:0] sample_mem [0:N_SAMPLES-1];

    initial begin
        $dumpfile(`VCD_FILE);
        /*for (idx = 0; idx < (R)*C; idx = idx + 1) begin 
            $dumpvars(0, dut.x_buf.uut.ram_block[idx]); 
            end
        */
        for (idx = 0; idx< C; idx = idx +1) $dumpvars(0, dut.b_0.Vt[idx]);
            
        for (idx = 0; idx< R; idx = idx +1) $dumpvars(0, dut.b_0.Us[idx]);
        
        $dumpvars(0);

        $readmemh("../python/samples.hex", sample_mem);

        $display("===== SVD Filter Testbench Start =====");

        // ── Test 1: Reset Behaviour ──────────────
        $display("\n[TEST 1] Reset check");
        apply_reset();
        wait_cycles(10000);
        if (y === 0)
            $display("PASS: y = 0 after reset");
        else
            $display("FAIL: y = %0d after reset (expected 0)", y);

        
        //for ( i=0 ;i <80*81 ; i++)begin
        //    send_sample(0);
        //end

        x_clk_en = 1;
        for ( i = 0; i < N_SAMPLES; i++) begin
        send_sample(sample_mem[i]);
        end

        wait_cycles(1000);
        $display("\n===== Testbench Complete =====");
        $finish;
    end

endmodule