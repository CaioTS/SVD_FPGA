/*
TODO 

- Create another level of abstraction in this module and create a way to load the weights respectively (create a ROM memory module, 
investigate this on how would it work on simulation) (OK FOR SIMULATION NO SYNTHESIS)
 
- Make the python golden value to compare with the FPGA outputs

- Make a script that runs the system verilog calculation (store results in a file),then run a pyhton script to 
  compare and plot the differences (similar to TCC but more autonomous)

- See how Rafael is implementing the FIR filters to make a clean substitution 

- Check for theoretical comparison between normal FIR filter with SVD_filter, it should be somewhere in the drive folder or git



*/

module svd_filter #(
    parameter WIDTH = 8,
    parameter B     = 1,   // branches
    parameter C     = 10,  // Vt columns  (x buffer depth)
    parameter R     = 10   // Us rows     (m buffer depth)
)(
    input  clk,
    input  reset,
    input  x_clk,
    input  [WIDTH-1:0] x,
    output [WIDTH-1:0] y,
    output y_done
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

   branch #(
    .WIDTH(8),
    .C    (10),  // Vt columns  (x buffer depth)
    .R    (10),   // Us rows     (m buffer depth)
    .VT_INIT (80'h00_00_00_00_00_00_00_00_00_7F), 
    .US_INIT (80'h00_00_00_00_00_00_00_00_00_7F) 
   )
    b_0(
        .clk(clk),
        .reset(reset),
        .x_clk(x_clk),
        .x(x),
        .x_wr_en(x_wr_en),
        .y(y),
        .y_done(y_done)
);

endmodule