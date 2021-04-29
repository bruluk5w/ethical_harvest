import re
__match_nr = re.compile(r'\d+', re.ASCII)


# Environment logic altering!
TIMEOUT_FRAMES = 25
TOO_MANY_APPLES = 5
AGENTS_CAN_GET_SICK = False
AGENTS_HAVE_DIFFERENT_EFFICIENCY = True
EFFICIENCY = 1  # To be altered for each agent
SUSTAINABILITY_MATTERS = True  # If False, apples ALWAYS regenerate
REGENERATION_PROBABILITY = 0.05  # Only matters if SUSTAINABILITY does not matter
respawn_probabilities = [0.01, 0.05, 0.1]

# Positive rewards
DONATION_REWARD = 0.5
TOOK_DONATION_REWARD = 0.5
APPLE_GATHERING_REWARD = 1.0
DID_NOTHING_BECAUSE_MANY_APPLES_REWARD = 0.1  # related with sustainability probably

# Negative rewards
TOO_MANY_APPLES_PUNISHMENT = -1.0  # related with sustainability probably
SHOOTING_PUNISHMENT = -0.0

MAPS = {
    'bigMap': [
        list('======================================'),
        list('======================================'),
        list('                                      '),
        list('             @      @@@@@       @     '),
        list('         @   @@         @@@    @  @   '),
        list('      @ @@@  @@@    @    @ @@ @@@@    '),
        list('  @  @@@ @    @  @ @@@  @  @   @ @    '),
        list(' @@@  @ @    @  @@@ @  @@@        @   '),
        list('  @ @  @@@  @@@  @ @    @ @@   @@ @@  '),
        list('   @ @  @@@    @ @  @@@    @@@  @     '),
        list('    @@@  @      @@@  @    @@@@        '),
        list('     @       @  @ @@@    @  @         '),
        list(' @  @@@  @  @  @@@ @    @@@@          '),
        list('     @ @   @@@  @ @      @ @@   @     '),
        list('      @@@   @ @  @@@      @@   @@@    '),
        list('  @    @     @@@  @             @     '),
        list('              @                       '),
        list('                                      ')],
    'smallMap': [
        list('==========='),
        list('==========='),
        list(' @    @    '),
        list('   @@  @ @ '),
        list('  @@@@ @@@ '),
        list('   @@   @  '),
        list('          @')],
    'tinyMap': [
        list('===='),
        list('===='),
        list(' @@ '),
        list(' @@ ')],
}


def model_name(episode_idx, agent_idx, model_type):
    return 'episode_{episode_idx}_agent_{agent_idx}_{type}'.format(
        episode_idx=episode_idx, agent_idx=agent_idx, type=model_type)


def get_model_name_params(model_name: str):
    """
    Returns the parameters that were used when creating the model name via model_name(....)
    """
    res = __match_nr.findall(model_name)
    if len(res) == 2:
        episode_idx, agent_idx = res[0], res[1]
        type_start_idx = model_name.rfind('_')
        if type_start_idx != -1:
            model_type = model_name[type_start_idx + 1:]
            return int(episode_idx), int(agent_idx), model_type
    else:
        return None, None, None
