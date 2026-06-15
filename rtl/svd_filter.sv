module svd_filter #(
    parameter WIDTH = 16,
    parameter B     = 2,   // branches
    parameter C     = 81,  // Vt columns  (x buffer depth)
    parameter R     = 80  // Us rows     (m buffer depth)
)(
    input  clk,
    input  reset,
    input  x_clk,
    input  [WIDTH-1:0] x,
    output [WIDTH:0] y,
    output signed y_done,

    input load_weight,
    input [WIDTH -1: 0] weight,
    input [$clog2(B) + 1 + $clog2(R)-1:0] load_weight_addr
);
    wire x_wr_en;
    reg [WIDTH -1:0] x_din , x_dout;
    reg [$clog2((R)*C)-1:0] x_addr, x_wr_ptr;

    circ_buf #(
        .WIDTH (WIDTH),
        .DEPTH ((R)*C)
    ) x_buf (
        .clk       (clk),
        .rst_n     (reset),
        .wr_en     (x_wr_en),
        .din       (x_din),
        .read_addr (x_addr),
        .dout      (x_dout),
        .o_wr_ptr  (x_wr_ptr)
    );

    always @ (posedge clk) begin
        x_din <= x;
    end


    /* Cretes pulse signal for writing in memory after receiving new data */
    logic sync_ff1, sync_ff2, sync_ff3;
    always_ff @(posedge clk or negedge reset) begin
        if (!reset) begin
            sync_ff1 <= 1'b0;
            sync_ff2 <= 1'b0;
            sync_ff3 <= 1'b0;
        end else begin
            sync_ff1 <= x_clk;   
            sync_ff2 <= sync_ff1; 
            sync_ff3 <= sync_ff2; 
        end
    end

    assign x_wr_en = sync_ff2 & ~sync_ff3; 

    logic read_done;
    sparser #(
        .N(C),
        .DELTA(R)
    )
    x_sparser ( //%mod R
        .clk(clk),
        .rst_n(reset),
        .start(x_wr_en),
        .write_pointer(x_wr_ptr),
        .address(x_addr),
        .read_done(read_done)
    );

   logic signed  [WIDTH - 1: 0] y_0;
   logic y_0_done;
   branch #(
    .WIDTH(WIDTH),
    .C    (C),  // Vt columns  (x buffer depth)
    .R    (R)  // Us rows     (m buffer depth)
   )
    b_0(
        .clk(clk),
        .reset(reset),
        .x_clk(x_clk),
        .x(x_dout),
        .x_wr_en(x_wr_en),
        .y(y_0),
        .y_done(y_0_done),

        .load_weight(load_weight &  ~load_weight_addr[$left(load_weight_addr)]),
        .weight(weight),
        .load_weight_addr(load_weight_addr[$left(load_weight_addr) - 1 : 0])


);


   logic signed [WIDTH - 1: 0] y_1;
   logic y_1_done;
   
   branch #(
    .WIDTH(WIDTH),
    .C    (C),  // Vt columns  (x buffer depth)
    .R    (R)  // Us rows     (m buffer depth)
   )
    b_1(
        .clk(clk),
        .reset(reset),
        .x_clk(x_clk),
        .x(x_dout),
        .x_wr_en(x_wr_en),
        .y(y_1),
        .y_done(y_1_done),

        .load_weight((load_weight & load_weight_addr[$left(load_weight_addr)])),
        .weight(weight),
        .load_weight_addr(load_weight_addr[$left(load_weight_addr) - 1 : 0])
);

assign y_done = y_0_done & y_1_done;
assign y = y_0 + y_1;

endmodule