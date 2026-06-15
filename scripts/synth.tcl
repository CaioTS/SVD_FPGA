# ── Project settings ──────────────────────────────────────────────
set PART    "xc7z007sclg225-2"
set TOP     "svd_filter"    
set RTL_DIR "../rtl"
set OUT_DIR "./output"

# Design settings for SVD filter
set WIDTH 16
set B      2
set C      64
set R      101

# Design setting for FIR filter
#set WIDTH 16
#set B      2
#set C      80
#set R      80

# Tool settings
#set opts "-flatten_hierarchy none -max_dsp 0"
set opts "-flatten_hierarchy none"

file mkdir $OUT_DIR

# ── Read sources ──────────────────────────────────────────────────
read_verilog -sv [glob $RTL_DIR/*.sv]

# ── Read constraints ──────────────────────────────────────────────
read_xdc ../constraints/Zybo-Z7-Master.xdc

# ── Synthesis ─────────────────────────────────────────────────────
synth_design -top $TOP -part $PART \
-generic WIDTH=$WIDTH \
-generic B=$B \
-generic C=$C \
-generic R=$R \
{*}$opts
#synth_design -top $TOP -part $PART -flatten_hierarchy none

# ── Reports ───────────────────────────────────────────────────────
report_timing_summary -file $OUT_DIR/timing_synth.rpt
report_utilization    -file $OUT_DIR/utilization_synth.rpt

# ── Write checkpoint ──────────────────────────────────────────────
write_checkpoint -force $OUT_DIR/post_synth.dcp

puts "Synthesis done."
