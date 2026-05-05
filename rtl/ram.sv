module ram#(
    parameter WIDTH = 8,
    parameter DEPTH = 10
)(
    input clk,
    input write_enable,
    input [$clog2(DEPTH)-1:0]address,
    input [WIDTH-1:0]data_in,
    output reg [WIDTH-1:0]data_out
);

reg [WIDTH - 1:0]ram_block[0:DEPTH-1];

always @(posedge clk) begin
        if(write_enable)
            ram_block[address] <= data_in;
        else
            data_out <= ram_block[address];
end

endmodule