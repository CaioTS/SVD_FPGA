/*
TODO 
 
- Make the python golden value to compare with the FPGA outputs

- Make a script that runs the system verilog calculation (store results in a file),then run a pyhton script to 
  compare and plot the differences (similar to TCC but more autonomous)

- See how Rafael is implementing the FIR filters to make a clean substitution 

- Check for theoretical comparison between normal FIR filter with SVD_filter, it should be somewhere in the drive folder or git



*/

module svd_filter #(
    parameter WIDTH = 8,
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

parameter [647:0] VT_ROW0 = 648'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FF000100FF0002FEFA0535;
parameter [639:0] US_ROW0 = 640'hEF001D1FF3E0011606D8D9060EF2CEE6190EECDA04310EECF0213C01E8012D2FE3DE092610C1DB1217F2B1EC260CE4BC133E01E5D93C4AECE8F45139C5E8054C1299F01035E9810F1F1DD48B46302001;
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


parameter [647:0] VT_ROW1 = 648'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100FF000100FE000200FD000500F9010AFDF00931FC;
parameter [639:0] US_ROW1 = 640'hF9ECEEF8FAEFE9F300FFF5F6061009000616180A030C170FFCF9040AFAE9EEFDFDEBE2F301FCECEF061105F8041C1A08FF0F2112FBF90C14F9E6F00402E4DCF405FBE0E8091100EC02221B03F7162B0E;
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