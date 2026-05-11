`timescale 1ns/1ps

module multiplier_tb;

    // ─────────────────────────────────────────────
    // Parameters
    // ─────────────────────────────────────────────
    localparam N          = 4;
    localparam WIDTH      = 8;
    localparam GUARD      = 4;
    localparam CLK_PERIOD = 10;

    // ─────────────────────────────────────────────
    // DUT Signals
    // ─────────────────────────────────────────────
    logic                  clk;
    logic                  rst_n;
    logic                  start;
    logic [WIDTH-1:0]      A [0:N-1];   // unpacked array for DUT port
    logic [WIDTH-1:0]      X;
    logic [WIDTH-1:0]      Y;
    logic                  calc_done;

    // ─────────────────────────────────────────────
    // DUT Instantiation
    // ─────────────────────────────────────────────
    multiplier #(
        .N     (N),
        .WIDTH (WIDTH),
        .GUARD (GUARD)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .A         (A),
        .X         (X),
        .Y         (Y),
        .calc_done (calc_done)
    );

    // ─────────────────────────────────────────────
    // Clock Generation
    // ─────────────────────────────────────────────
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ─────────────────────────────────────────────
    // Helpers — flat packed vectors for task args
    // iverilog does not support unpacked array ports
    // in tasks/functions, so we pass a flat vector
    // [N*WIDTH-1:0] and slice with +: inside.
    //
    // Packing convention (MSB → LSB):
    //   flat[N*WIDTH-1 -: WIDTH] = A[N-1]
    //   ...
    //   flat[WIDTH-1   -: WIDTH] = A[0]
    // ─────────────────────────────────────────────

    // Load flat vector into DUT unpacked array A
    task automatic load_coeffs(input logic [N*WIDTH-1:0] flat);
        for (int i = 0; i < N; i++)
            A[i] = flat[i*WIDTH +: WIDTH];
    endtask

    // Software dot product for checking (returns raw pre-scale integer)
    function automatic integer expected_dot(
        input logic [N*WIDTH-1:0] flat,
        input logic [WIDTH-1:0]   x_val
    );
        integer acc;
        acc = 0;
        for (int i = 0; i < N; i++)
            acc = acc + ($signed(flat[i*WIDTH +: WIDTH]) * $signed(x_val));
        expected_dot = acc;
    endfunction

    // ─────────────────────────────────────────────
    // Task : apply_reset
    // ─────────────────────────────────────────────
    task automatic apply_reset();
        $display("[%0t] >> RESET asserted", $time);
        rst_n = 0;
        start = 0;
        X     = '0;
        for (int i = 0; i < N; i++) A[i] = '0;
        repeat(4) @(posedge clk); #1;
        rst_n = 1;
        @(posedge clk); #1;
        $display("[%0t] >> RESET released", $time);
    endtask

    // ─────────────────────────────────────────────
    // Task : run_multiply
    // flat_coeff : N*WIDTH packed vector
    //              flat[i*WIDTH +: WIDTH] = A[i]
    // x_val      : input sample
    // tag        : display label
    // ─────────────────────────────────────────────
    task automatic run_multiply(
        input logic [N*WIDTH-1:0] flat_coeff,
        input logic [WIDTH-1:0]   x_val,
        input string              tag
    );
        integer exp;
        integer got;

        // Load coefficients and input sample
        load_coeffs(flat_coeff);
        X = x_val;

        // Pulse start for one cycle
        @(posedge clk); #1;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        // Wait for DUT to finish
        @(posedge calc_done); #1;

        exp = expected_dot(flat_coeff, x_val);
        got = $signed(Y);

        $display("[%0t] %s", $time, tag);
        $display("         X        = %0d (0x%02h)", $signed(x_val), x_val);
        for (int i = 0; i < N; i++)
            $display("         A[%0d]    = %0d", i, $signed(flat_coeff[i*WIDTH +: WIDTH]));
        $display("         Y (DUT)  = %0d (0x%02h)", got, Y);
        $display("         dot(A,X) = %0d (pre-scale)", exp);

        // Let FSM return to IDLE before next test
        @(posedge clk); #1;
    endtask

    // ─────────────────────────────────────────────
    // Waveform dump
    // ─────────────────────────────────────────────
    initial begin
        $dumpfile(`VCD_FILE);
        $dumpvars(0, multiplier_tb);
    end

    // ─────────────────────────────────────────────
    // Timeout watchdog
    // ─────────────────────────────────────────────
    initial begin
        #(CLK_PERIOD * 5000);
        $display("[%0t] WATCHDOG TIMEOUT", $time);
        $finish;
    end

    // ─────────────────────────────────────────────
    // Test Sequence
    //
    // Flat coefficient format for N=4, WIDTH=8:
    //   32'hA3_A2_A1_A0  where A[i] = flat[i*8 +: 8]
    //   so A[0]=A0, A[1]=A1, A[2]=A2, A[3]=A3
    // ─────────────────────────────────────────────
    initial begin
        apply_reset();

        // ── Test 1: All-ones ──────────────────────
        // A=[1,1,1,1], X=4 → dot = 16
        $display("\n[TEST 1] All-ones coefficients");
        run_multiply(32'h01_01_01_01, 8'd4, "A=[1,1,1,1] X=4");
        
        // ── Test 2: Single non-zero coefficient ───
        // A=[1,0,0,0], X=127 → dot = 127
        $display("\n[TEST 2] Single non-zero coefficient");
        run_multiply(32'h00_00_00_01, 8'd127, "A=[1,0,0,0] X=127");

        // ── Test 3: Signed coefficients ───────────
        // A=[-1,-1,-1,127], X=1 → dot = 124
        $display("\n[TEST 3] Mixed signed coefficients");
        run_multiply(32'hFF_FF_FF_7F, 8'd1, "A=[127,-1,-1,-1] X=1");

        // ── Test 4: Zero input ────────────────────
        // A=[50,50,50,50], X=0 → dot = 0
        $display("\n[TEST 4] Zero input");
        run_multiply(32'h32_32_32_32, 8'd0, "A=[50,50,50,50] X=0");

        // ── Test 5: Back-to-back computations ─────
        $display("\n[TEST 5] Back-to-back computations");
        run_multiply(32'h02_02_02_02, 8'd10, "A=[2,2,2,2] X=10 (1st)");
        run_multiply(32'h02_02_02_02, 8'd20, "A=[2,2,2,2] X=20 (2nd)");

        // ── Test 6: Max positive values ───────────
        // A=[127,127,127,127], X=127 → dot = 4*127*127 = 64516
        $display("\n[TEST 6] Max positive saturation");
        run_multiply(32'h7F_7F_7F_7F, 8'd127, "A=[127,127,127,127] X=127");

        // ── Test 7: Negative X ────────────────────
        // A=[1,1,1,1], X=-1 (0xFF) → dot = -4
        $display("\n[TEST 7] Negative input X=-1");
        run_multiply(32'h01_01_01_01, 8'hFF, "A=[1,1,1,1] X=-1");
        
        $display("\n===== Testbench Complete =====");
        $finish;
    end

endmodule