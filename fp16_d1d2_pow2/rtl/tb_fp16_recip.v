// Testbench for fp16_recip: exhaustively drives all 1024 FP16 inputs in
// [1,2) and checks the integer result against the real 1/x reference.
//
// Reference: y*2^11 = 2048 / (1 + frac/1024), rounded to nearest.
// The polynomial error (0.45 ulp at d1=3,d2=7) plus output rounding allows
// |y_q11 - ref| <= 2 LSB.
`timescale 1ns / 1ps

module tb_fp16_recip;

    reg         clk = 0;
    reg         rst_n = 0;
    reg         in_valid = 0;
    reg  [9:0]  frac = 0;
    wire        out_valid;
    wire [11:0] y_q11;
    wire [15:0] fp16_out;

    fp16_recip dut (
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .frac(frac),
        .out_valid(out_valid), .y_q11(y_q11), .fp16_out(fp16_out)
    );

    always #5 clk = ~clk;

    integer i, errors, max_diff, diff;
    integer got_y [0:1023];
    integer got_v [0:1023];
    integer ref_i;
    real    x_r, ref_q11;

    initial begin
        errors = 0;
        max_diff = 0;
        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;

        // stream all 1024 inputs, one per cycle (in-order pipeline)
        for (i = 0; i < 1024; i = i + 1) begin
            @(negedge clk);
            frac = i[9:0];
            in_valid = 1;
        end
        @(negedge clk);
        in_valid = 0;

        // drain pipeline: outputs are collected by order below
        wait (out_valid);
        // capture: pipeline latency is fixed; collect 1024 results
        // (handled by the capture block)
        repeat (8) @(posedge clk);
        $display("TB done: %0d errors, max |diff| = %0d LSB", errors, max_diff);
        if (errors == 0) $display("PASS");
        else $display("FAIL");
        $finish;
    end

    // capture block: FIFO by order
    integer wr = 0;
    always @(posedge clk) begin
        if (out_valid) begin
            x_r = 1.0 + (wr * 1.0) / 1024.0;
            ref_q11 = 2048.0 / x_r;
            ref_i = $rtoi(ref_q11 + 0.5);          // round to nearest (positive)
            diff = y_q11 - ref_i;
            if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 2) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH frac=%0d y_q11=%0d ref=%0d diff=%0d",
                             wr, y_q11, ref_i, diff);
            end
            if (y_q11 < 1024 || y_q11 > 2048) begin
                errors = errors + 1;
                $display("RANGE frac=%0d y_q11=%0d", wr, y_q11);
            end
            wr = wr + 1;
        end
    end

endmodule
