from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    ForcedEOSTokenLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    TemperatureLogitsWarper,
    LogitsProcessorList
)

text = "abc\ndef\n"
num = text.count("\n")

print("End")
