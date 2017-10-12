"""
This script writes out prediction given a trained model

Usage:
python mythic_writer_character.py --model_file ./foo.pt --seed_string Bar

@author: Brad Beechler (brad.e.beechler@gmail.com)
# Last Modification: 09/20/2017 (Brad Beechler)
"""
from uplog import log
import torch
import mythic_common as common
import mythic_model_character
from torch.autograd import Variable


def generate(net, seed_string='A', predict_length=100, temperature=0.8, cuda=False):
    hidden = net.init_hidden(1)
    if cuda:
        prime_input = Variable(common.char_tensor(seed_string).unsqueeze(0)).cuda()
    else:
        prime_input = Variable(common.char_tensor(seed_string).unsqueeze(0))

    # Use seed string to "build up" hidden state
    predicted = seed_string
    for p in range(len(seed_string) - 1):
        _, hidden = net(prime_input[:, p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_length):
        output, hidden = net(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = common.trainable_characters[top_i]
        predicted += predicted_char
        inp = mythic_model_character.Variable(common.char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted


if __name__ == '__main__':
    """
    CLI driver to rerun a model prediction
    """
    # Get settings from command line
    write_settings = common.WriterSettings()
    if write_settings.debug:
        log.out.setLevel('DEBUG')
    else:
        log.out.setLevel('INFO')

    # Parse command line arguments
    log.out.info("Loading model from file: " + write_settings.model_file)
    net = torch.load(write_settings.model_file)

    # This is just to make hacking on experiments a bit easier
    seed_string = write_settings.seed_string
    temperature = write_settings.temperature

    # Set up output file if requested
    if write_settings.output_file is not None:
        outfile = open(write_settings.output_file, 'w')

    # Run a loop
    for i in range(write_settings.iterations):
        predicted_string = generate(net,
                                    seed_string=seed_string,
                                    predict_length=write_settings.predict_length,
                                    temperature=temperature,
                                    cuda=write_settings.cuda)
        log.out.info("Seed string: " + "\n" + write_settings.seed_string)
        log.out.info("Predicted string: " + "\n" + predicted_string)
        outfile.write("%s\n" % predicted_string)
