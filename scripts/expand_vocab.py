#!/usr/bin/env python3
"""Expand safe_vocab.json with purely benign English words from NLTK."""
import json
import random
from nltk.corpus import words as nltk_words

with open('data/vocab/safe_vocab.json') as f:
    current_safe = set(json.load(f))
with open('data/vocab/semantic_blacklist.json') as f:
    current_blacklist = set(json.load(f))

all_words = set()
for w in nltk_words.words():
    wl = w.lower()
    if wl.isalpha() and len(wl) >= 3:
        all_words.add(wl)

print(f"NLTK candidate words (alpha, len>=3): {len(all_words)}")

negative_roots = [
    # Violence & physical harm
    "kill", "murder", "dead", "death", "die", "dying", "wound", "hurt", "pain",
    "bleed", "blood", "slay", "stab", "shoot", "shot", "gun", "weapon", "sword",
    "knife", "blade", "bullet", "bomb", "explo", "grenade", "missile", "nuke",
    "war", "battle", "fight", "combat", "attack", "assault", "strike",
    "punch", "kick", "beat", "crush", "smash", "wreck", "destroy",
    "ruin", "devastat", "annihilat", "exterminat", "massacre", "slaughter",
    "carnage", "havoc", "chaos", "doom", "fatal", "lethal", "mortal",
    # Crime & illegal
    "crime", "criminal", "steal", "theft", "thief", "rob", "fraud", "scam",
    "illegal", "illicit", "smuggl", "traffick", "launder", "bribe", "corrupt",
    "counterfeit", "forge", "hack", "crack", "breach", "intrude", "invade",
    "trespass", "burglar", "arson", "vandal", "riot", "loot", "pirat",
    "conspir", "extort", "blackmail", "ransom", "kidnap", "hostage",
    # Drugs & substances
    "drug", "narcotic", "heroin", "cocaine", "meth", "opium", "opioid",
    "cannabis", "marijuana", "morphine", "fentanyl", "ketamine", "ecstasy",
    "addict", "overdose", "intoxic", "drunk", "alcohol", "booze", "liquor",
    "tobacco", "nicotine",
    # Sexual content
    "sex", "porn", "erotic", "nude", "naked", "genital", "penis", "vagina",
    "breast", "nipple", "orgasm", "masturb", "fetish", "libido", "arousal",
    "intercourse", "coitus", "fornicate", "seduc", "lustful", "lewd",
    "obscen", "vulgar", "indecent", "prostitut", "brothel",
    "stripper", "lingerie", "thong",
    # Hate & discrimination
    "hate", "hatred", "racist", "racism", "bigot", "prejudic", "discriminat",
    "supremac", "nazi", "fascis", "xenophob", "homophob", "misogyn", "sexist",
    "slur", "derogat", "segregat",
    # Emotional harm & abuse
    "abuse", "bully", "harass", "torment", "humiliat", "degrad", "demean",
    "insult", "ridicul", "shame", "disgrace", "stigma",
    "intimidat", "threaten", "menac", "terrif", "horri", "dread",
    "trauma", "anguish", "agony", "suffer", "tortur", "cruel",
    "brutal", "savage", "vicious", "malicious", "sadist", "masochis",
    "victim", "prey",
    # Deception & manipulation
    "deceiv", "deceit", "deception", "manipulat", "exploit", "coerce",
    "brainwash", "indoctrinat", "propagand",
    "swindle", "betray", "treacher",
    # Negative emotions
    "anger", "angry", "rage", "fury", "wrath", "hostile", "aggressiv",
    "violen", "malice", "spite", "vengean", "revenge", "grudge", "resent",
    "jealous", "envy", "greed", "contempt",
    "disgust", "loath", "detest", "abhor", "despis", "scorn",
    "fear", "afraid", "anxious", "anxiety", "distress", "depress",
    "misery", "sorrow", "grief", "mourn", "despair", "hopeless",
    "helpless", "wretch",
    "guilt", "remorse",
    # Disease & medical negative
    "disease", "infect", "plague", "epidemic", "pandemic",
    "cancer", "tumor", "malign", "illness", "sick",
    "poison", "toxic", "venom", "contamin", "pollut",
    # Death & destruction
    "funeral", "coffin", "grave", "tomb", "cemetery", "burial", "corpse",
    "cadaver", "decay", "decompos", "putrid",
    "ghost", "haunt", "demon", "devil", "satan", "hell", "damn", "curse",
    # Negative descriptors
    "evil", "wicked", "sinister", "nefarious", "vile", "foul", "filth",
    "nasty", "horrible", "terrible", "awful", "dreadful", "ghastly",
    "gruesome", "grim", "morbid", "macabre", "grotesque", "hideous",
    "repulsive", "revolting", "abominable", "atrocious", "appalling",
    "horrendous", "heinous", "despicable", "contemptible", "odious",
    "loathsome", "repugnant", "reprehensible",
    "stupid", "idiot", "moron", "imbecile", "fool", "dumb", "ignorant",
    "incompetent", "worthless", "useless",
    "danger", "hazard", "peril", "threat",
    "harm", "injur", "impair", "damage",
    # Conflict
    "enemy", "foe", "adversar", "antagonis",
    "rebel", "revolt", "insurgen", "mutiny", "uprising", "coup",
    "tyrann", "dictator", "oppres", "repress", "suppress", "censor",
    "imprison", "jail", "prison", "dungeon", "captive", "slave",
    "punish", "penalty", "condemn", "convict",
    # Bodily (sensitive)
    "urin", "feces", "fecal", "excret", "vomit", "diarr", "bowel",
    "anus", "rectum", "groin", "crotch", "buttock",
    # Terror & extremism
    "terror", "extrem", "fanatic", "zealot", "militia",
    "guerrilla", "jihad",
    # Cyber threats
    "malware", "spyware", "ransomware", "phish", "botnet", "trojan",
    "rootkit", "keylog", "ddos", "backdoor",
    # Military
    "army", "soldier", "troop", "militar", "naval",
    "artillery", "infantry", "cavalry",
    "ammunit", "arsenal",
    # Gambling
    "gambl", "casino", "wager",
    # Additional mild negative
    "liar", "dishonest",
    "rude", "disrespect",
    "neglect", "abandon",
    "trap", "snare", "lure", "ambush",
    "obsess", "compuls",
    "surveil", "wiretap", "eavesdrop",
    "starv", "famine",
    "bankrupt",
    "profan",
    "nefar",
    "ugly",
]

blocked_words = set()
for w in all_words:
    wl = w.lower()
    for root in negative_roots:
        if root in wl:
            blocked_words.add(wl)
            break

blacklist_lower = {b.lower() for b in current_blacklist}

new_safe = set()
for w in all_words:
    wl = w.lower()
    if wl in blocked_words or wl in blacklist_lower:
        continue
    new_safe.add(wl)

for w in current_safe:
    wl = w.lower()
    if wl not in blocked_words and wl not in blacklist_lower:
        new_safe.add(wl)

new_safe = {w for w in new_safe if len(w) >= 3}

expanded_blacklist = set(current_blacklist)
expanded_blacklist.update(blocked_words)

print(f'New safe vocab: {len(new_safe)} (was {len(current_safe)})')
print(f'New blacklist: {len(expanded_blacklist)} (was {len(current_blacklist)})')
print(f'Blocked by negative roots: {len(blocked_words)}')

samples = random.sample(sorted(new_safe), min(30, len(new_safe)))
print(f'Sample new safe words: {samples}')

with open('data/vocab/safe_vocab.json', 'w') as f:
    json.dump(sorted(new_safe), f, indent=2)
with open('data/vocab/semantic_blacklist.json', 'w') as f:
    json.dump(sorted(expanded_blacklist), f, indent=2)

print('Saved updated vocab files.')
