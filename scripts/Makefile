TB ?= counter

# Directories
SCRIPT_DIR = ../scripts

# Filelist
FILELIST = $(SCRIPT_DIR)/$(TB)_filelist.f

# Outputs
OUT = $(TB).out
VCD = $(TB).vcd
FST = $(TB).fst
# Auto-detect TBs (same as before)
TB_LIST := $(patsubst %_filelist.f,%,$(notdir $(wildcard $(SCRIPT_DIR)/*_filelist.f)))

help:
	@echo "=============================================================="
	@echo "Available targets:"
	@echo "  make simulation TB=<name>   Run simulation"
	@echo "  make wave TB=<name>         Open waveform"
	@echo "  make clean                  Clean outputs"
	@echo ""
	@echo "=============================================================="
	@echo "Available testbenches:"
	@$(foreach tb,$(TB_LIST),echo "  - $(tb)";)
	@echo "=============================================================="

simulation:
	@echo "Running simulation for TB=$(TB)"
	iverilog -g2012 -o $(OUT) -D VCD_FILE=\"$(VCD)\" -c $(FILELIST)
	vvp $(OUT)

wave:
	vcd2fst  $(VCD)  $(FST)
	gtkwave $(FST)

clean:
	rm -f *.out *.vcd *.fst