module sparser #(
    parameter N = 10, //Number of samples needed for request
    parameter DELTA = 10 //Number of samples to jump each time
)(
    input  clk,   //main clk higher frequency
    input  rst_n,
    input  start, //Write or read enable (synchronous pulse) (will ac)
    input  [$clog2((DELTA) *N)-1:0] write_pointer, //Comes from memory (should be registered before write is complete)
    output reg [$clog2((DELTA) *N)-1:0] address,
    output reg read_done
);

   reg [$clog2((DELTA) *N)-1:0] write_pointer_reg ;
   reg wait_ok;
   reg [$clog2(N-1):0 ] sample_counter;
    typedef enum logic [2:0] {
        RESET = 3'b000,
        IDLE  = 3'b001,
        START = 3'b010,
        WAIT_WRITE = 3'b011,
        CALCULATE_ADDRESS = 3'b100,
        READ = 3'b101,
        DONE = 3'b110
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
                    next_state = IDLE;
            end
 
            IDLE: begin
                if (start) next_state = START;
                else next_state = IDLE;
            end
 
            START: begin
                next_state = WAIT_WRITE;
            end

            WAIT_WRITE: begin
                if (wait_ok) next_state = CALCULATE_ADDRESS;
                else next_state = WAIT_WRITE; 
            end

            CALCULATE_ADDRESS:  begin
                next_state = READ;
            end

            READ: begin
                if (sample_counter == 0) next_state = DONE;
                else next_state = CALCULATE_ADDRESS;
            end
            DONE: begin
                next_state = IDLE;
            end
            default: next_state = RESET; 
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wait_ok <= 0;
            address  <= 0;
            write_pointer_reg <= 0;
            sample_counter <= 0;
            read_done <= 0;
        end 
        else begin
            wait_ok <= 0;
            read_done <= 0;

            case (current_state)
            RESET: begin
                address  <= 0;
                write_pointer_reg <= 0;
            end
 
            IDLE: begin
                address <= 0;
                write_pointer_reg <= 0;
            end
 
            START: begin
                address <= 0;
                write_pointer_reg <= write_pointer;

            end

            WAIT_WRITE: begin
                sample_counter <= N-1;
                address <= write_pointer_reg;
                wait_ok <= 1;
            end

            CALCULATE_ADDRESS:  begin
                write_pointer_reg <= write_pointer_reg;
                sample_counter <= sample_counter - 1'b1;
                if (address >= (DELTA))
                    address <= address - (DELTA);
                else
                    address <= address - (DELTA) + ((DELTA)*N);
                end

            READ: begin
                //Wait for Memory to display the output calculated by the address
                sample_counter <= sample_counter;
                address <= address;
            end
            DONE: begin
                read_done <= 1;
            end
        endcase
        end
    end

endmodule
