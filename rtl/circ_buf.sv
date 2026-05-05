
/*
The idea here will be to write in memory always as a circular buffer. 
But reading can be done isolated. 
So that teh matrix calculation state machine operates, it only send signals to read when needed, 
only using one ouput at a time. 

A nice check is to see if i can read the data stored from any address using two clock_cycles, 
and if writing is really circular.
*/

module circ_buf #(
    parameter WIDTH = 8,
    parameter DEPTH = 10
)(
    input  clk,   //main clk higher frequency
    input  rst_n,
    input  [$clog2(DEPTH)-1:0] read_addr, 
    input  wr_en, //Write or read enable (synchronous pulse)
    input  [WIDTH-1:0] din, //Data to write
    output reg  [WIDTH-1:0] dout,  //Data read
    output [$clog2(DEPTH)-1:0] o_wr_ptr
);

    reg [$clog2(DEPTH)-1:0] wr_ptr,ram_ptr;
    reg [WIDTH -1 : 0] ram_din;
    reg ram_wr_en;
    integer i;


    ram #(
    .WIDTH(WIDTH),
    .DEPTH(DEPTH)
    ) uut (
    .clk          (clk),
    .write_enable (ram_wr_en),
    .address      (ram_ptr),
    .data_in      (ram_din),
    .data_out     (dout)
    );

    /*
    STATES: RESET, READ , WRITE
    */

    typedef enum logic [1:0] {
        RESET = 2'b00,
        READ  = 2'b01,
        WRITE = 2'b10
    } state_t;
 
    state_t current_state, next_state;
 
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            current_state <= RESET;
        else
            current_state <= next_state;
    end


     always_comb begin
        next_state = RESET;
        case (current_state)
            RESET: begin                
                    next_state = READ;
            end
 
            READ: begin
                if (wr_en) next_state = WRITE;
                else next_state = READ;
            end
 
            WRITE: begin
                next_state = READ;
            end
 
            default: next_state = RESET; 
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ram_ptr <= 0;
            wr_ptr  <= 0;
            ram_wr_en <= 0;
            ram_din <= 0;
        end 
        
        else begin
            ram_din <= din;
            case (current_state)
                RESET: begin
                    ram_ptr <= read_addr;
                    wr_ptr <= 0;
                    ram_wr_en <= 0;                
                end
    
                READ: begin
                    ram_ptr <= read_addr;
                    wr_ptr <= wr_ptr;
                    ram_wr_en <= 0;
                end
    
                WRITE: begin
                    ram_ptr <= wr_ptr;
                    wr_ptr  <= (wr_ptr + 1) % DEPTH;
                    ram_wr_en <= 1; 
                end
            endcase
        end
    end


assign o_wr_ptr = wr_ptr;
endmodule
