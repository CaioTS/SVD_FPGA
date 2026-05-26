module branch #(
    parameter WIDTH = 8,
    parameter C     = 10,  // Vt columns  (x buffer depth)
    parameter R     = 10,   // Us rows     (m buffer depth)
    parameter [C*WIDTH-1:0] VT_INIT = '0,
    parameter [R*WIDTH-1:0] US_INIT = '0
)(
    input  clk,
    input  reset,
    input  x_clk,
    input  [WIDTH-1:0] x,
    input  x_wr_en,
    output [WIDTH-1:0] y,
    output y_done
);
 logic start;
 assign start = x_wr_en & coeff_loaded;
 logic VTx_calc_done;
    wire [WIDTH - 1 : 0]Vtx_y;

reg signed [WIDTH-1:0] Vt [0:C-1];   // right-singular vectors  (C coefficients)
reg signed [WIDTH-1:0] Us [0:R-1];   // left-singular * sigma   (R coefficients)


    multiplier #(
        .N(C),
        .WIDTH(WIDTH),
        .GUARD(0)
    ) VTx
    (
        .clk(clk),
        .rst_n(reset),
        .start(start),
        .A(Vt),
        .X(x),
        .Y(Vtx_y),
        .calc_done(VTx_calc_done)
    );  

    reg [WIDTH -1:0] u_din , u_dout;
    reg [$clog2((R))-1:0] u_addr, u_wr_ptr;

    circ_buf #(
        .WIDTH (WIDTH),
        .DEPTH ((R))
    ) u_buf (
        .clk       (clk),
        .rst_n     (reset),
        .wr_en     (VTx_calc_done),
        .din       (Vtx_y),
        .read_addr (u_addr),
        .dout      (u_dout),
        .o_wr_ptr  (u_wr_ptr)
    );
 
    logic u_read_done;
    sparser #(
        .N(R),
        .DELTA(1)
    )
    u_sparser (
        .clk(clk),
        .rst_n(reset),
        .start(VTx_calc_done),
        .write_pointer(u_wr_ptr),
        .address(u_addr),
        .read_done(u_read_done)
    );
    multiplier #(
        .N(R),
        .WIDTH(WIDTH),
        .GUARD(0)
    ) US_VTX
    (
        .clk(clk),
        .rst_n(reset),
        .start(VTx_calc_done),
        .A(Us),
        .X(u_dout),
        .Y(y),
        .calc_done(y_done)
    );  


typedef enum logic {
        COEFF_LOADING = 1'b0,
        COEFF_DONE    = 1'b1
    } coeff_state_t;
 
    coeff_state_t coeff_state;
 
    // Use the wider of C and R for the counter
    localparam COEFF_CNT_WIDTH = $clog2((C > R ? C : R)) + 1;
 
    logic [COEFF_CNT_WIDTH-1:0] coeff_cnt;
    logic                       coeff_loaded;  // high when coefficients are ready
 
    always_ff @(posedge clk or negedge reset) begin
        if (!reset) begin
            coeff_state  <= COEFF_LOADING;
            coeff_cnt    <= '0;
            coeff_loaded <= 1'b0;
            for (int i = 0; i < C; i++) Vt[i] <= '0;
            for (int i = 0; i < R; i++) Us[i] <= '0;
        end
        else begin
            case (coeff_state)
 
                COEFF_LOADING: begin
                    // Phase 1: fill Vt (indices 0..C-1)
                    if (coeff_cnt < COEFF_CNT_WIDTH'(C)) begin
                        Vt[coeff_cnt] <= VT_INIT[coeff_cnt*WIDTH +: WIDTH];
                        coeff_cnt     <= coeff_cnt + 1'b1;
                    end
                    // Phase 2: fill Us (indices 0..R-1), reuse counter offset by C
                    else if (coeff_cnt < COEFF_CNT_WIDTH'(C + R)) begin
                        Us[coeff_cnt - COEFF_CNT_WIDTH'(C)] <= US_INIT[(coeff_cnt - COEFF_CNT_WIDTH'(C))*WIDTH +: WIDTH];
                        coeff_cnt <= coeff_cnt + 1'b1;
                    end
                    // Both arrays loaded
                    else begin
                        coeff_state  <= COEFF_DONE;
                        coeff_loaded <= 1'b1;
                    end
                end
 
                COEFF_DONE: begin
                    coeff_loaded <= 1'b1;   // hold high until next reset
                end
 
            endcase
        end
    end

endmodule