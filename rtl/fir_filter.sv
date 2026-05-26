module fir_filter #(
    parameter WIDTH = 16,
    parameter B     = 2,   // branches
    parameter C     = 80,  // Vt columns  (x buffer depth)
    parameter R     = 80   // Us rows     (m buffer depth)
)(
    input  clk,
    input  reset,
    input  x_clk,
    input  [WIDTH-1:0] x,
    output [WIDTH-1:0] y,
    output signed y_done
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
    reg signed [WIDTH-1:0] W [0:(C*R)-1];   // left-singular * sigma   (R coefficients)

    sparser #(
        .N(C*R),
        .DELTA(1)
    )
    x_sparser (
        .clk(clk),
        .rst_n(reset),
        .start(x_wr_en),
        .write_pointer(x_wr_ptr),
        .address(x_addr),
        .read_done(read_done)
    );

    multiplier #(
        .N(C*R),
        .WIDTH(WIDTH),
        .GUARD(0)
    ) FIR_mult
    (
        .clk(clk),
        .rst_n(reset),
        .start(x_wr_en),
        .A(W),
        .X(x_dout),
        .Y(y),
        .calc_done(y_done)
    );  




initial $readmemh("../weights/FIR_weights.hex",W);
endmodule