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

// ── Vt and Us coefficient matrices ──────────
    reg [WIDTH-1:0] Vt [0:C-1];   // right-singular vectors  (C coefficients)
    reg [WIDTH-1:0] Us [0:R-1];   // left-singular * sigma   (R coefficients)

    //TODO This does not work for synthesys, only for simulation. By now use this for testing if the module is working
    //, later make a proper interface to load these values. Maybe load these in reset.
    integer i ;
    initial begin
        for (int i = 0; i < C; i++)
            Vt[i] = VT_INIT[i*WIDTH +: WIDTH];
        for (int i = 0; i < R; i++)
            Us[i] = US_INIT[i*WIDTH +: WIDTH];
    end

 logic VTx_calc_done;
    wire [WIDTH - 1 : 0]Vtx_y;
    multiplier #(
        .N(C),
        .WIDTH(WIDTH),
        .GUARD(0)
    ) VTx
    (
        .clk(clk),
        .rst_n(reset),
        .start(x_wr_en),
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

endmodule