/*
TODO 
 
- See how Rafael is implementing the FIR filters to make a clean substitution 

- Check for theoretical comparison between normal FIR filter with SVD_filter, it should be somewhere in the drive folder or git

*/

module svd_filter #(
    parameter WIDTH = 16,
    parameter B     = 2,   // branches
    parameter C     = 81,  // Vt columns  (x buffer depth)
    parameter R     = 80   // Us rows     (m buffer depth)
)(
    input  clk,
    input  reset,
    input  x_clk,
    input  [WIDTH-1:0] x,
    output [WIDTH:0] y,
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
    x_sparser (
        .clk(clk),
        .rst_n(reset),
        .start(x_wr_en),
        .write_pointer(x_wr_ptr),
        .address(x_addr),
        .read_done(read_done)
    );

parameter [1295:0] VT_ROW0 = 1296'h0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FFFF000000010000FFFF00000002FFFEFFFA00050035;
parameter [1279:0] US_ROW0 = 1280'hFFEF0000001D001FFFF3FFE0000100160006FFD8FFD90006000EFFF2FFCEFFE60019000EFFECFFDA00040031000EFFECFFF00021003C0001FFE80001002D002FFFE3FFDE000900260010FFC1FFDB00120017FFF2FFB1FFEC0026000CFFE4FFBC0013003E0001FFE5FFD9003C004AFFECFFE8FFF400510039FFC5FFE80005004C0012FF99FFF000100035FFE9FF81000F001F001DFFD4FF8B0046003000200001;
   logic signed  [WIDTH - 1: 0] y_0;
   logic y_0_done;
   branch #(
    .WIDTH(WIDTH),
    .C    (C),  // Vt columns  (x buffer depth)
    .R    (R),   // Us rows     (m buffer depth)
    .VT_INIT (VT_ROW0), 
    .US_INIT (US_ROW0) 
   )
    b_0(
        .clk(clk),
        .reset(reset),
        .x_clk(x_clk),
        .x(x_dout),
        .x_wr_en(x_wr_en),
        .y(y_0),
        .y_done(y_0_done)
);


parameter [1295:0] VT_ROW1 = 1296'h0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000FFFF000000010000FFFE000000020000FFFD000000050000FFF90001000AFFFDFFF000090031FFFC;
parameter [1279:0] US_ROW1 = 1280'hFFF9FFECFFEEFFF8FFFAFFEFFFE9FFF30000FFFFFFF5FFF60006001000090000000600160018000A0003000C0017000FFFFCFFF90004000AFFFAFFE9FFEEFFFDFFFDFFEBFFE2FFF30001FFFCFFECFFEF000600110005FFF80004001C001A0008FFFF000F00210012FFFBFFF9000C0014FFF9FFE6FFF000040002FFE4FFDCFFF40005FFFBFFE0FFE8000900110000FFEC00020022001B0003FFF70016002B000E;
   logic signed [WIDTH - 1: 0] y_1;
   logic y_1_done;
   
   branch #(
    .WIDTH(WIDTH),
    .C    (C),  // Vt columns  (x buffer depth)
    .R    (R),   // Us rows     (m buffer depth)
    .VT_INIT (VT_ROW1), 
    .US_INIT (US_ROW1) 
   )
    b_1(
        .clk(clk),
        .reset(reset),
        .x_clk(x_clk),
        .x(x_dout),
        .x_wr_en(x_wr_en),
        .y(y_1),
        .y_done(y_1_done)
);

assign y_done = y_0_done & y_1_done;
assign y = y_0 + y_1;

endmodule