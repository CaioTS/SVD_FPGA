module branch #(
    parameter WIDTH = 8,
    parameter C     = 10,  // Vt columns  (x buffer depth)
    parameter R     = 10   // Us rows     (m buffer depth)
)(
    input  clk,
    input  reset,
    input  x_clk,
    input  [WIDTH-1:0] x,
    input  x_wr_en,
    output [WIDTH-1:0] y,
    output y_done,

    input load_weight,
    input [WIDTH -1: 0] weight,
    input [1 + $clog2(R)-1:0] load_weight_addr
);
    logic start;
    assign start = x_wr_en;
    logic VTx_calc_done;
    wire [WIDTH - 1 : 0]Vtx_y;

    multiplier #(
        .N(C),
        .WIDTH(WIDTH),
        .GUARD(0)
    ) 
    VTx (
        .clk(clk),
        .rst_n(reset),
        .start(start),
        .X(x),
        .Y(Vtx_y),
        .calc_done(VTx_calc_done),


        .load_weight(load_weight & ~load_weight_addr[$left(load_weight_addr)]),
        .weight(weight),
        .load_weight_addr(load_weight_addr[$clog2(C)-1 : 0])
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
        //.A(Us),
        .X(u_dout),
        .Y(y),
        .calc_done(y_done),
        

        .load_weight(load_weight & load_weight_addr[$left(load_weight_addr)]),
        .weight(weight),
        .load_weight_addr(load_weight_addr[$clog2(R)-1 : 0])    
        );  


endmodule