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

#Vivado related paths
VIVADO_SRC  = /opt/Xilinx/2025.2/Vivado/settings64.sh
VIVADO      = vivado
SCRIPT_DIR  = ../scripts
SIM_DIR     = sim
OUT_DIR     = output
FST         = $(SIM_DIR)/waves.fst
TCL_SCRIPT  = $(SCRIPT_DIR)/gtkwave_setup.tcl

SOURCES = $(wildcard rtl/*.sv)

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

synth:
	#source $(VIVADO_SRC)
	mkdir -p $(OUT_DIR)
	$(VIVADO) -mode batch -source $(SCRIPT_DIR)/synth.tcl \
	          -log $(OUT_DIR)/synth.log \
	          -journal $(OUT_DIR)/synth.jou
clean:
	rm -rf output
	rm -rf *.out *.vcd *.fst *.txt *.jou *.log 
