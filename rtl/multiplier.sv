module multiplier #(
    parameter N     = 10,  // Number of samples (taps)
    parameter X_WIDTH = 8,
    parameter W_WIDTH = 8,
    parameter Y_WIDTH = 8,
    parameter GUARD = $clog2(N)    // log2(N) guard bits to prevent accumulator overflow
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    start,
    input  logic signed [X_WIDTH-1:0]        X,            // Input sample — hold stable for full computation
    output logic signed [Y_WIDTH-1:0]        Y,            // Filter output (upper WIDTH bits of accumulator)
    output logic                    calc_done,

    input load_weight,
    input [$clog2(N)-1:0] load_weight_addr,
    input [W_WIDTH-1:0 ]weight
);

    logic signed [W_WIDTH-1: 0] A;
    // -------------------------------------------------------------------------
    // Internal registers
    // -------------------------------------------------------------------------
    logic [1:0]                    wait_cnt;     
    logic [$clog2(N)-1:0]          tap_index;
    logic [$clog2(N)-1:0]          ram_address;    
    logic signed [W_WIDTH -1:  0]  a_reg;
    logic signed [X_WIDTH -1 : 0]  x_reg;
    logic signed [(X_WIDTH + W_WIDTH)-1:0]     product;
    logic signed [(X_WIDTH + W_WIDTH + GUARD) -1:0] accumulator;
    logic signed [(X_WIDTH + W_WIDTH + GUARD) -1:0] y_reg;
    // -------------------------------------------------------------------------
    // State encoding
    // -------------------------------------------------------------------------
    typedef enum logic [2:0] {
        RESET       = 3'b000,
        IDLE        = 3'b001,
        START       = 3'b010,
        WAIT_CYCLES = 3'b011, 
        LOAD_ACC    = 3'b100,
        MULTIPLY    = 3'b101,
        DONE        = 3'b110
    } state_t;

    state_t current_state, next_state;

    // -------------------------------------------------------------------------
    // State register
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            current_state <= RESET;
        else
            current_state <= next_state;
    end

    // -------------------------------------------------------------------------
    // Next-state logic
    // -------------------------------------------------------------------------
    always_comb begin
    case (current_state)
        RESET:       next_state = IDLE;

        IDLE: begin
            if (start) next_state = START;
            else        next_state = IDLE;
        end

        START:       next_state = WAIT_CYCLES;

        WAIT_CYCLES: begin
            if (wait_cnt == 2'b10) next_state = LOAD_ACC;
            else                   next_state = WAIT_CYCLES;
        end

        LOAD_ACC:    next_state = MULTIPLY;

        MULTIPLY: begin
            if (tap_index == $clog2(N)'(N)) next_state = DONE;
            else                               next_state = LOAD_ACC;
        end

        DONE:        next_state = IDLE;

        default:     next_state = RESET;
    endcase
end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wait_cnt    <= '0;
            tap_index   <= '0;
            a_reg      <= '0;
            x_reg       <= '0;
            product     <= '0;
            accumulator <= '0;
            y_reg       <= '0;
            calc_done   <= 1'b0;
        end
        else begin
            calc_done <= 1'b0; 

            case (current_state)
                RESET: begin
                    wait_cnt    <= '0;
                    tap_index   <= '0;
                    accumulator <= '0;
                    product     <= '0;
                    y_reg       <= '0;
                end

                IDLE: begin
                    wait_cnt    <= '0;
                    tap_index   <= '0;
                    accumulator <= '0;
                    product     <= '0;
                    x_reg <= '0;
                    a_reg <= '0;  
                end

                START: begin
                    wait_cnt <= '0;
                    x_reg <= '0;
                    a_reg <= '0;   
                end

                WAIT_CYCLES: begin
                    wait_cnt  <= wait_cnt + 1'b1;
                    tap_index <= '0;
                end


                LOAD_ACC: begin
                    x_reg    <= X;  
                    a_reg      <= A;
                    tap_index   <= tap_index + 1'b1;
                    accumulator <= accumulator + product;  
                end

                MULTIPLY: begin
                    product <= x_reg * a_reg;
                end

                DONE: begin
                    y_reg <= accumulator + product;
                    calc_done <= 1'b1;
                end

            endcase
        end
    end

assign Y = y_reg[(X_WIDTH + W_WIDTH + GUARD) -1 -: Y_WIDTH];


assign ram_address = load_weight ? load_weight_addr : tap_index;
ram #(
    .WIDTH (W_WIDTH),
    .DEPTH (N)
    )  VT_w(
    .clk(clk),
    .write_enable(load_weight),
    .address(ram_address),
    .data_in(weight),
    .data_out(A)
    );

endmodule