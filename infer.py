from logging import getLogger, StreamHandler, DEBUG, INFO
import cmd
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.logging import set_verbosity_info
# set up logging
root_logger = getLogger()
logger = getLogger(__name__)
logger.setLevel(INFO)
handler = StreamHandler()
handler.setLevel(DEBUG)
root_logger.addHandler(handler)
set_verbosity_info()


# Create a simple CMD parser to interact with the model and set parameters
class InferenceShell(cmd.Cmd):
    prompt: str = ">> "
    intro: str = "Welcome to the model inference shell.\n" \
                 "Type 'help' to see a list of available commands.\n" \
                 "Type 'exit' to quit the shell.\n"

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.min_length = 0
        self.max_length = 0
        super().__init__()

    def do_max_length(self, line):
        """ Set the maximum length of the generated text """
        self.max_length = int(line)

    def do_min_length(self, line):
        """ Set the minimum length of the generated text """
        self.min_length = int(line)

    def do_exit(self, line):
        return True

    def do_infer(self, line):
        """ Run inference on a single sentence. """
        inputs = self.tokenizer(line, return_tensors="pt")
        args = {}
        if self.min_length:
            args["min_length"] = self.min_length
        if self.max_length:
            args["max_length"] = self.max_length
        outputs = self.model.generate(inputs["input_ids"].to(0), **args)
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    def do_hf_map(self, line):
        """ Print the device map """
        print(self.model.hf_device_map)

    def do_hf_map_offload(self, line):
        """ Print the offload folder """
        print(self.model.hf_device_map_offload)

    def do_hf_map_offload_size(self, line):
        """ Print the size of the offload folder """
        print(self.model.hf_device_map_offload_size)

    def do_hf_map_offload_free(self, line):
        """ Print the free space of the offload folder """
        print(self.model.hf_device_map_offload_free)

    # If the command is unrecognized assume the user is trying to talk to the model
    def default(self, line):
        self.do_infer(line)


if __name__ == '__main__':
    model_path = 'H:/bloom'
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", offload_folder='offload', torch_dtype=torch.bfloat16)
    print(model.hf_device_map)
    logger.info ("Starting inference shell")
    shell  = InferenceShell(tokenizer, model)
    shell .cmdloop()
