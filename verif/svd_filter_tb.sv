`timescale 1ns/1ps

module svd_filter_tb;

    // ─────────────────────────────────────────────
    // Parameters
    // ─────────────────────────────────────────────
    parameter WIDTH = 8;
    parameter B     = 1;
    parameter C     = 10;
    parameter R     = 10;

    // ─────────────────────────────────────────────
    // DUT Signals
    // ─────────────────────────────────────────────
    reg                  clk;
    reg                  reset;
    reg                  x_clk;
    reg  [WIDTH-1:0]     x;
    wire [WIDTH-1:0]     y;

    integer idx;

    // ─────────────────────────────────────────────
    // Clock Generation
    // clk   : 10ns period (50 MHz)
    // x_clk :  period (400 Hz) — Much Slower
    // ─────────────────────────────────────────────
    initial clk = 0;
    always #5  clk   = ~clk; 

    initial x_clk = 0;
    always #5000 x_clk = ~x_clk;

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
        .y     (y)
    );

    // ─────────────────────────────────────────────
    // Tasks
    // ─────────────────────────────────────────────

    // Apply reset
    task apply_reset;
        begin
            reset = 0;          // active low
            x     = 0;
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

    initial begin
        $dumpfile(`VCD_FILE);
        for (idx = 0; idx < (R)*C; idx = idx + 1) begin 
            $dumpvars(0, dut.x_buf.uut.ram_block[idx]); 
            end
        $dumpvars(0);


        $display("===== SVD Filter Testbench Start =====");

        // ── Test 1: Reset Behaviour ──────────────
        $display("\n[TEST 1] Reset check");
        apply_reset();
        wait_cycles(4);
        if (y === 0)
            $display("PASS: y = 0 after reset");
        else
            $display("FAIL: y = %0d after reset (expected 0)", y);

        
        for ( i=0 ;i <100 ; i++)begin
            send_sample(0);
        end

        //for ( i=0 ;i <150 ; i++)begin
        //    send_sample(WIDTH'(i));
        //end


        send_sample(WIDTH'(10));

        wait_cycles(1000);
        $display("\n===== Testbench Complete =====");
        $finish;
    end

endmodule