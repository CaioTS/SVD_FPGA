# ── Project settings ──────────────────────────────────────────────
set PART    "xc7z007sclg225-2"
set TOP     "svd_filter"    
set RTL_DIR "../rtl"
set OUT_DIR "./output"

file mkdir $OUT_DIR

# ── Read sources ──────────────────────────────────────────────────
read_verilog -sv [glob $RTL_DIR/*.sv]

# ── Read constraints ──────────────────────────────────────────────
read_xdc ../constraints/Zybo-Z7-Master.xdc

# ── Synthesis ─────────────────────────────────────────────────────
synth_design -top $TOP -part $PART

# ── Reports ───────────────────────────────────────────────────────
report_timing_summary -file $OUT_DIR/timing_synth.rpt
report_utilization    -file $OUT_DIR/utilization_synth.rpt

# ── Write checkpoint ──────────────────────────────────────────────
write_checkpoint -force $OUT_DIR/post_synth.dcp

puts "Synthesis done."
