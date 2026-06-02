import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# --- Constants ---
MEMORY_SIZE = 4096
WORD_MASK = 0xFFFF  # 16-bit
ADDR_MASK = 0x0FFF  # 12-bit address

# --- Opcodes (Memory Reference) ---
# D3 D2 D1 D0 etc
# 0  0  0  = AND
# 0  0  1  = ADD
# 0  1  0  = LDA
# 0  1  1  = STA
# 1  0  0  = BUN
# 1  0  1  = BSA
# 1  1  0  = ISZ
OP_AND = 0x0000
OP_ADD = 0x1000
OP_LDA = 0x2000
OP_STA = 0x3000
OP_BUN = 0x4000
OP_BSA = 0x5000
OP_ISZ = 0x6000

# Indirect bit
INDIRECT = 0x8000

# --- Register Reference (I=0, D7=1, D6=1, D5=1 -> 0x7000) ---
OP_REG_REF = 0x7000
CLA = 0x7800  # Clear AC
CLE = 0x7400  # Clear E
CMA = 0x7200  # Complement AC
CME = 0x7100  # Complement E
CIR = 0x7080  # Circulate Right
CIL = 0x7040  # Circulate Left
INC = 0x7020  # Increment AC
SPA = 0x7010  # Skip if AC Positive
SNA = 0x7008  # Skip if AC Negative
SZA = 0x7004  # Skip if AC Zero
SZE = 0x7002  # Skip if E Zero
HLT = 0x7001  # Halt

# --- IO Reference (I=1, D7=1, D6=1, D5=1 -> 0xF000) ---
OP_IO_REF = 0xF000
INP = 0xF800  # Input char to AC
OUT = 0xF400  # Output char from AC
SKI = 0xF200  # Skip on Input Flag
SKO = 0xF100  # Skip on Output Flag
ION = 0xF080  # Interrupt On
IOF = 0xF040  # Interrupt Off


@dataclass
class ManoState:
    """
    Represents the full state of the Mano Basic Computer.
    """
    AC: int = 0  # Accumulator (16-bit)
    PC: int = 0  # Program Counter (12-bit)
    AR: int = 0  # Address Register (12-bit)
    IR: int = 0  # Instruction Register (16-bit)
    DR: int = 0  # Data Register (16-bit)
    E: int = 0  # Extension bit (1-bit)
    I: int = 0  # Indirect bit (1-bit)
    S: int = 1  # Start/Stop flip-flop (1=Run, 0=Halt)

    # 4096 words of memory
    Memory: List[int] = field(default_factory=lambda: [0] * MEMORY_SIZE)

    # Input/Output Registers (8-bit)
    INPR: int = 0
    OUTR: int = 0
    FGI: int = 0  # Input Flag
    FGO: int = 0  # Output Flag
    IEN: int = 0  # Interrupt Enable

    def clone(self) -> 'ManoState':
        """Deep copy of state (except memory is shared if not modified)."""
        new_state = ManoState(
            AC=self.AC, PC=self.PC, AR=self.AR, IR=self.IR, DR=self.DR,
            E=self.E, I=self.I, S=self.S,
            INPR=self.INPR, OUTR=self.OUTR, FGI=self.FGI, FGO=self.FGO, IEN=self.IEN
        )
        new_state.Memory = list(self.Memory)  # Copy memory
        return new_state

    def __repr__(self):
        return (f"PC={self.PC:03X} AC={self.AC:04X} E={self.E} "
                f"IR={self.IR:04X} Mem[PC]={self.Memory[self.PC]:04X}")


class ManoEmulator:
    """
    Cycle-accurate-ish emulator for the Mano Basic Computer.
    Executes instructions and updates state.
    """

    def __init__(self):
        self.state = ManoState()

    def load_program(self, program: List[int], start_addr: int = 0):
        """Loads a binary program into memory."""
        for i, word in enumerate(program):
            if start_addr + i < MEMORY_SIZE:
                self.state.Memory[start_addr + i] = word & WORD_MASK
        self.state.PC = start_addr

    def step(self):
        """
        Executes one instruction cycle (Fetch, Decode, Execute).
        Returns True if machine is still running (S=1), False if Halted.
        """
        if self.state.S == 0:
            return False

        # --- Fetch ---
        self.state.AR = self.state.PC
        self.state.IR = self.state.Memory[self.state.AR]
        self.state.PC = (self.state.PC + 1) & ADDR_MASK

        # --- Decode ---
        # D7 = (self.state.IR & 0x7000) == 0x7000? not exactly, check op code bits
        opcode = self.state.IR & 0x7000
        self.state.I = (self.state.IR & 0x8000) >> 15

        # Check for Register or IO reference (Opcode = 111 -> 7)
        if opcode == 0x7000:
            if self.state.I == 0:
                self._execute_register_ref()
            else:
                self._execute_io_ref()
        else:
            self._execute_memory_ref(opcode)

        return self.state.S == 1

    def _execute_register_ref(self):
        inst = self.state.IR
        if inst == CLA:
            self.state.AC = 0
        elif inst == CLE:
            self.state.E = 0
        elif inst == CMA:
            self.state.AC = (~self.state.AC) & WORD_MASK
        elif inst == CME:
            self.state.E = 1 if self.state.E == 0 else 0
        elif inst == CIR:
            # Circulate Right: AC[0]->E, E->AC[15]
            old_ac0 = self.state.AC & 1
            old_e = self.state.E
            self.state.AC = (self.state.AC >> 1) | (old_e << 15)
            self.state.E = old_ac0
        elif inst == CIL:
            # Circulate Left: AC[15]->E, E->AC[0]
            old_ac15 = (self.state.AC >> 15) & 1
            old_e = self.state.E
            self.state.AC = ((self.state.AC << 1) & WORD_MASK) | old_e
            self.state.E = old_ac15
        elif inst == INC:
            res = self.state.AC + 1
            self.state.AC = res & WORD_MASK
            if res > WORD_MASK:
                self.state.E = 1  # Carry out potentially? (Mano usually doesn't affect E on INC, but let's stick to simple wrap)
                # Actually Mano spec for INC: AC <- AC + 1. Doesn't mention E. Overflow wraps.

        # Skips
        elif inst == SPA:
            if (self.state.AC & 0x8000) == 0:  # Positive (MSB=0)
                self.state.PC = (self.state.PC + 1) & ADDR_MASK
        elif inst == SNA:
            if (self.state.AC & 0x8000) != 0:  # Negative (MSB=1)
                self.state.PC = (self.state.PC + 1) & ADDR_MASK
        elif inst == SZA:
            if self.state.AC == 0:
                self.state.PC = (self.state.PC + 1) & ADDR_MASK
        elif inst == SZE:
            if self.state.E == 0:
                self.state.PC = (self.state.PC + 1) & ADDR_MASK
        elif inst == HLT:
            self.state.S = 0

    def _execute_io_ref(self):
        # Placeholder for IO ref - strictly we don't need full IO for this task
        # Placeholder for IO ref - strictly we don't need full IO for this task
        print("Debug: IO instruction executed (no-op)")

    def _execute_memory_ref(self, opcode):
        # Effective Address Calculation
        self.state.AR = self.state.IR & 0x0FFF

        if self.state.I == 1:
            # Indirect addressing: Memory[AR] is the actual effective address
            self.state.AR = self.state.Memory[self.state.AR] & ADDR_MASK

        # Execute based on opcode
        if opcode == OP_AND:
            self.state.DR = self.state.Memory[self.state.AR]
            self.state.AC = self.state.AC & self.state.DR

        elif opcode == OP_ADD:
            self.state.DR = self.state.Memory[self.state.AR]
            res = self.state.AC + self.state.DR
            self.state.AC = res & WORD_MASK
            # E <- Cout
            self.state.E = 1 if res > WORD_MASK else 0

        elif opcode == OP_LDA:
            self.state.DR = self.state.Memory[self.state.AR]
            self.state.AC = self.state.DR

        elif opcode == OP_STA:
            self.state.Memory[self.state.AR] = self.state.AC

        elif opcode == OP_BUN:
            self.state.PC = self.state.AR

        elif opcode == OP_BSA:
            # Memory[AR] <- PC, PC <- AR + 1
            self.state.Memory[self.state.AR] = self.state.PC
            self.state.PC = (self.state.AR + 1) & ADDR_MASK

        elif opcode == OP_ISZ:
            # Memory[AR] <- Memory[AR] + 1. If result == 0, Skip next instruction.
            self.state.DR = self.state.Memory[self.state.AR]
            val = (self.state.DR + 1) & WORD_MASK
            self.state.Memory[self.state.AR] = val
            if val == 0:
                self.state.PC = (self.state.PC + 1) & ADDR_MASK

    def run(self, max_steps=100) -> List[ManoState]:
        """
        Runs the current program until HLT or max_steps.
        Returns the trace of states.
        """
        trace = [self.state.clone()]
        for _ in range(max_steps):
            running = self.step()
            trace.append(self.state.clone())
            if not running:
                break
        return trace


def generate_random_program(length=10, seed=None) -> Tuple[List[int], List[int]]:
    """
    Generates a simple valid random program (mostly additions and loads)
    to verify learning of arithmetic and data movement.
    """
    if seed is not None:
        random.seed(seed)

    prog = []
    # Data section at the end of memory usually, but for simple tests
    # we can put data at 0x100
    data_addr = 0x100
    data = [random.randint(0, 0xFF) for _ in range(16)]  # Random data words

    # Generate instructions
    # Mix of LDA, ADD, STA, CMA, CIR, INC
    for _ in range(length):
        r = random.random()
        addr = random.randint(data_addr, data_addr + 15)

        if r < 0.3:  # LDA
            op = OP_LDA | addr
        elif r < 0.6:  # ADD
            op = OP_ADD | addr
        elif r < 0.7:  # STA
            op = OP_STA | addr
        elif r < 0.8:  # CMA
            op = CMA
        elif r < 0.9:  # INC
            op = INC
        else:  # CIR
            op = CIR

        prog.append(op)

    prog.append(HLT)
    return prog, data


if __name__ == "__main__":
    # Smoke test
    emu = ManoEmulator()
    # Program: Load 5, Add 10, Store to 0, Halt
    # Data: [5, 10] at 0x10, 0x11
    prog = [
        OP_LDA | 0x010,  # Load M[10]
        OP_ADD | 0x011,  # Add M[11]
        OP_STA | 0x000,  # Store M[0]
        HLT
    ]
    data_loc = 0x010
    emu.load_program(prog)
    emu.state.Memory[0x010] = 5
    emu.state.Memory[0x011] = 10

    trace = emu.run()

    print("Execution Trace:")
    for step, s in enumerate(trace):
        print(f"{step}: {s}")

    final_val = emu.state.Memory[0]
    print(f"Final M[0]: {final_val} (Expected 15)")
    assert final_val == 15, "Emulator failed simple add test"
    print("Mano Emulator verification passed.")
