module rom #(
    parameter WIDTH  = 16,
    parameter DEPTH  = 6400,
    parameter W_FILE = "W.hex"
)(
    input  logic                      clk,
    input  logic [$clog2(DEPTH)-1:0]  addr,
    output logic signed [WIDTH-1:0]   dout
);

(* ram_style = "distributed" *) reg signed [WIDTH-1:0] mem [0:DEPTH-1];

    initial $readmemh(W_FILE, mem);

    always_ff @(posedge clk)
        dout <= mem[addr];

endmodule