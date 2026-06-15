`timescale 1ns/1ps

module fir_filter_tb;

    // ─────────────────────────────────────────────
    // Parameters
    // ─────────────────────────────────────────────
    parameter WIDTH = 16;
    parameter B     = 2;
    parameter C     = 80;
    parameter R     = 80;

    // ─────────────────────────────────────────────
    // DUT Signals
    // ─────────────────────────────────────────────
    reg                  clk;
    reg                  reset;
    reg                  x_clk;
    reg x_clk_en;
    reg  signed [WIDTH-1:0]     x;
    wire signed [WIDTH-1:0]     y;

    integer idx;

    reg load_weight;
    reg signed [WIDTH -1: 0] weight;
    reg [$clog2((R)*C)-1:0] load_weight_addr;
    // ─────────────────────────────────────────────
    // Clock Generation
    // clk   : 10ns period (50 MHz)
    // x_clk :  period (400 Hz) — Much Slower
    // ─────────────────────────────────────────────
    initial clk = 0;
    always #5  clk   = ~clk; 

    initial x_clk = 0;
    initial x_clk_en = 0;
    always #80000 x_clk = (x_clk_en) ? ~x_clk : 0;

    // ─────────────────────────────────────────────
    // DUT Instantiation
    // ─────────────────────────────────────────────
    fir_filter
        #(
        .WIDTH(WIDTH),
        .B(B),
        .C(C),
        .R(R)
        )
         dut (
        .clk   (clk),
        .reset (reset),
        .x_clk (x_clk),
        .x     (x),
        .y     (y),
        .y_done(y_done),

        .load_weight(load_weight),
        .weight(weight),
        .load_weight_addr(load_weight_addr)
    );

    // ─────────────────────────────────────────────
    // Tasks
    // ─────────────────────────────────────────────

    logic signed [WIDTH-1:0] W_param [0:(R*C)-1];
    task load_weights;
        begin
        
        $readmemh("../weights/FIR_weights.hex", W_param);

        load_weight = 1'b1;

        #100us;
        @ (posedge clk);
        for (int i = 0; i< R*C ; i++) begin
            load_weight_addr <= i;
            weight <= W_param[i];
            @(posedge clk);
        end

        #100us;
        load_weight = 1'b0;
        @(posedge clk);
        end
    endtask
    // Apply reset
    task apply_reset;
        begin
            reset = 0;          // active low
            x     = 0;
            sample_idx = 0;
            ignore_first_sample = 1;
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
        if (!load_weight) begin
            if (!ignore_first_sample) ignore_first_sample = 1;
            else begin $display("Output y = %d  (x = %d)", y, sample_mem[sample_idx - 1]);
                $fdisplay(fd, "%0d, %0d, %0d", sample_idx - 1, sample_mem[sample_idx - 1], y);
            end
            sample_idx <= sample_idx + 1;
        end
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
        //for (idx = 0; idx< C; idx = idx +1) $dumpvars(0, dut.b_0.Vt[idx]);
            
        //for (idx = 0; idx< R; idx = idx +1) $dumpvars(0, dut.b_0.Us[idx]);
        
        $dumpvars(0);

        $readmemh("../python/samples.hex", sample_mem);

        $display("===== FIR Filter Testbench Start =====");

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

        load_weights();


        for ( i = 0; i < N_SAMPLES; i++) begin
        send_sample(sample_mem[i]);
        end

        wait_cycles(1000);
        $display("\n===== Testbench Complete =====");
        $finish;
    end

endmodule