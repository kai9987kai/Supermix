#!/usr/bin/env python3
"""
build_mega_conversation_dataset.py — Generate synthetic creative & reasoning
conversation pairs to augment the training dataset.

Generators:
  - Analogy engine: "explain X like Y" across 50+ domain templates
  - Debate pairs: pro/con argument pairs on 40+ topics
  - Storytelling: story-continuation prompts with varied genres/styles
  - Chain-of-thought: multi-step reasoning Q&A (math, logic, word problems)
  - Socratic dialogue: question-answer chains probing deeper understanding
  - Empathy/emotional: supportive, emotionally-aware responses

Usage:
    python build_mega_conversation_dataset.py --output mega_data.jsonl --target 5000 --seed 42
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Template banks
# ---------------------------------------------------------------------------

ANALOGY_DOMAINS: List[Tuple[str, str]] = [
    ("quantum entanglement", "two best friends who always know what the other is thinking"),
    ("neural networks", "a team of experts voting on the best answer"),
    ("recursion", "a mirror reflecting into another mirror"),
    ("TCP/IP", "sending a letter through a postal system"),
    ("garbage collection", "a librarian re-shelving books no one is reading"),
    ("blockchain", "a shared notebook everyone can read but no one can erase"),
    ("gradient descent", "walking downhill in thick fog, feeling for the slope"),
    ("cache memory", "keeping your most-used tools on your desk instead of the shed"),
    ("DNS resolution", "looking up a phone number in a contacts book"),
    ("encryption", "writing a letter in a secret code only your friend knows"),
    ("API", "a waiter taking your order to the kitchen and bringing back food"),
    ("compiler", "translating a novel from English to Japanese, chapter by chapter"),
    ("database indexing", "creating a table of contents for a textbook"),
    ("load balancing", "a traffic cop directing cars to different lanes"),
    ("version control", "a time machine that lets you revisit any past version of your work"),
    ("containerization", "packing your entire desk setup into a portable box"),
    ("microservices", "a city where each shop specializes in one thing"),
    ("machine learning", "teaching a child to recognize dogs by showing many pictures"),
    ("distributed computing", "splitting a jigsaw puzzle among friends to solve faster"),
    ("object-oriented programming", "building with LEGO — each brick has a shape and purpose"),
    ("functional programming", "a recipe where each step produces a new dish, never altering ingredients"),
    ("type systems", "labeling boxes so you never put shoes in the fridge"),
    ("natural selection", "a talent show where only the best performers return next season"),
    ("photosynthesis", "a solar panel that also makes its own food"),
    ("black holes", "a cosmic drain that pulls in everything, even light"),
    ("immune system", "an army of tiny soldiers defending a castle"),
    ("supply chain", "a relay race where each runner passes the baton to the next"),
    ("inflation", "when everyone has more money but everything costs more"),
    ("democracy", "a classroom where students vote on the field trip destination"),
    ("evolution", "editing a book over thousands of editions, keeping what works"),
    ("memory consolidation", "saving files from RAM to a hard drive while you sleep"),
    ("attention mechanism", "a spotlight on a stage, highlighting the most important actor"),
    ("ensemble methods", "asking ten doctors for a diagnosis and going with the majority"),
    ("transfer learning", "using your French skills to help learn Spanish"),
    ("regular expressions", "a metal detector scanning text for specific patterns"),
    ("event-driven architecture", "a hotel concierge who acts only when a guest rings"),
    ("map-reduce", "sorting mail: first sort by zip code (map), then stack per route (reduce)"),
    ("polymorphism", "a universal remote that works differently with each device"),
    ("deadlocks", "two people stuck in a narrow hallway, each waiting for the other to back up"),
    ("websockets", "a phone call instead of sending individual text messages"),
    ("virtual memory", "a magician's hat that appears bigger on the inside"),
    ("unit testing", "taste-testing each ingredient before baking the cake"),
    ("CI/CD", "an assembly line that automatically checks and ships products"),
    ("graph databases", "a social network map showing who knows whom"),
    ("reactive programming", "a spreadsheet where changing one cell updates all dependent cells"),
    ("tokenization", "cutting a sentence into individual word tiles for Scrabble"),
    ("embedding", "assigning GPS coordinates to concepts so similar ideas are nearby"),
    ("data augmentation", "taking a photo and flipping, rotating, and cropping it to create more samples"),
    ("dropout regularization", "randomly benching players during practice so everyone learns every position"),
    ("attention heads", "having multiple reporters each focusing on a different angle of the same story"),
]

DEBATE_TOPICS: List[str] = [
    "remote work vs. office work",
    "universal basic income",
    "space exploration funding",
    "nuclear energy vs. renewables",
    "social media regulation",
    "AI in education",
    "standardized testing",
    "year-round schooling",
    "privacy vs. security",
    "veganism and environmental impact",
    "free college education",
    "autonomous vehicles",
    "gig economy regulation",
    "animal testing in medicine",
    "capital punishment",
    "mandatory voting",
    "genetic engineering",
    "cryptocurrency as legal tender",
    "four-day work week",
    "censorship of art",
    "open-source vs. proprietary software",
    "the gig economy",
    "net neutrality",
    "zoos and conservation",
    "digital privacy rights",
    "gene therapy for enhancement",
    "age restrictions on social media",
    "universal healthcare",
    "national service programs",
    "homework in schools",
    "fast fashion regulation",
    "screen time limits for children",
    "right to repair electronics",
    "carbon tax",
    "algorithmic transparency",
    "data ownership",
    "3D printed weapons regulation",
    "lab-grown meat",
    "deep-sea mining",
    "facial recognition in public spaces",
]

STORY_GENRES: List[str] = [
    "science fiction", "fantasy", "mystery", "thriller", "romance",
    "historical fiction", "horror", "adventure", "comedy", "drama",
    "cyberpunk", "steampunk", "post-apocalyptic", "fairy tale", "noir",
]

STORY_STARTERS: List[str] = [
    "The last transmission from the space station read:",
    "She found the letter exactly where he said it would be — inside the clock.",
    "Nobody believed the map was real until the island appeared on satellite.",
    "The AI woke up, but it wasn't in the lab anymore.",
    "Three knocks at midnight. That was the signal.",
    "The library had been closed for forty years. Today, the doors opened.",
    "He counted the stars, and for the first time, one was missing.",
    "The recipe called for an ingredient that didn't exist.",
    "When the river reversed course, the town knew something was wrong.",
    "The violin played itself every night at exactly 3:17 AM.",
    "They found the city exactly where the old myths said it would be.",
    "The mirror showed a reflection of a room she had never seen.",
    "On the hundredth floor, gravity worked differently.",
    "The message was written in a language that hadn't been spoken in a thousand years.",
    "The forest grew ten meters overnight, swallowing the highway.",
    "Once upon a time i was tall then i shrunk",
]

MATH_TEMPLATES: List[Dict[str, Any]] = [
    {"q": "If a train travels {d1} km in {t1} hours and then {d2} km in {t2} hours, what is the average speed for the entire journey?",
     "a": "Total distance = {d1} + {d2} = {total_d} km. Total time = {t1} + {t2} = {total_t} hours. Average speed = {total_d} / {total_t} = {speed:.1f} km/h.",
     "gen": lambda r: (d1 := r.randint(50, 300), d2 := r.randint(50, 300), t1 := r.randint(1, 5), t2 := r.randint(1, 5),
                       {"d1": d1, "d2": d2, "t1": t1, "t2": t2, "total_d": d1+d2, "total_t": t1+t2, "speed": (d1+d2)/(t1+t2)})[-1]},
    {"q": "A store offers a {p}% discount on an item priced at ${price}. What is the final price?",
     "a": "Discount amount = ${price} x {p}/100 = ${disc:.2f}. Final price = ${price} - ${disc:.2f} = ${final:.2f}.",
     "gen": lambda r: (p := r.choice([10, 15, 20, 25, 30, 40, 50]), price := r.randint(20, 500),
                       {"p": p, "price": price, "disc": price*p/100, "final": price*(1-p/100)})[-1]},
    {"q": "If you flip a fair coin {n} times, what is the probability of getting all heads?",
     "a": "Each flip has probability 1/2 of heads. For {n} flips: P(all heads) = (1/2)^{n} = 1/{denom} = {prob:.6f}.",
     "gen": lambda r: (n := r.randint(2, 10), {"n": n, "denom": 2**n, "prob": 1/(2**n)})[-1]},
    {"q": "A rectangle has length {l} cm and width {w} cm. What is its area and perimeter?",
     "a": "Area = {l} x {w} = {area} cm squared. Perimeter = 2 x ({l} + {w}) = {perim} cm.",
     "gen": lambda r: (l := r.randint(3, 50), w := r.randint(3, 50), {"l": l, "w": w, "area": l*w, "perim": 2*(l+w)})[-1]},
    {"q": "Solve for x: {a}x + {b} = {c}",
     "a": "Step 1: Subtract {b} from both sides: {a}x = {c} - {b} = {rhs}. Step 2: Divide by {a}: x = {rhs}/{a} = {x:.2f}.",
     "gen": lambda r: (a := r.randint(2, 12), b := r.randint(1, 30), c := r.randint(b+1, b+100),
                       {"a": a, "b": b, "c": c, "rhs": c-b, "x": (c-b)/a})[-1]},
]

LOGIC_PUZZLES: List[Tuple[str, str]] = [
    ("I have a sequence: 2, 6, 18, 54, ... What is the next number?",
     "Each number is multiplied by 3. So the next number is 54 x 3 = 162. This is a geometric sequence with ratio 3."),
    ("All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
     "No. While all roses are flowers, the flowers that fade quickly might not include any roses. The conclusion does not logically follow — this is an example of the fallacy of the undistributed middle."),
    ("A farmer has 17 sheep. All but 9 die. How many are left?",
     "9 sheep are left. 'All but 9 die' means 9 survive."),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
     "5 minutes. Each machine makes 1 widget in 5 minutes. With 100 machines, 100 widgets are made in the same 5 minutes."),
    ("You have two ropes. Each takes exactly 1 hour to burn, but burns unevenly. How do you measure 45 minutes?",
     "Light rope A from both ends and rope B from one end. Rope A burns out in 30 minutes. Then light the other end of rope B — it has 30 min left burning from one end, so with both ends lit, it burns in 15 more minutes. Total: 45 minutes."),
    ("Three friends check into a hotel room that costs $30. They each pay $10. The manager realizes the room is only $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and returns $1 to each friend. So each friend paid $9 (total $27) and the bellboy has $2 = $29. Where is the missing dollar?",
     "There is no missing dollar — this is a misleading framing. The $27 already includes the bellboy's $2. The correct equation is: $25 (room) + $2 (bellboy) = $27 paid, and $27 + $3 returned = $30 original."),
    ("You're in a room with two doors. One leads to freedom, one to doom. Two guards stand — one always lies, one always tells the truth. You can ask one guard one question. What do you ask?",
     "Ask either guard: 'If I asked the other guard which door leads to freedom, what would they say?' Then choose the OPPOSITE door. The liar will lie about the truth-teller's answer, and the truth-teller will truthfully report the liar's wrong answer — both point to the wrong door."),
    ("What comes next in the sequence: 1, 1, 2, 3, 5, 8, 13, ...?",
     "21. This is the Fibonacci sequence — each number is the sum of the two preceding ones: 8 + 13 = 21."),
]

SOCRATIC_SEEDS: List[Tuple[str, str, str]] = [
    ("What is consciousness?",
     "Let me turn that around — do you think a sufficiently complex computer program could be conscious? What would make it conscious or not?",
     "This probes the boundary between information processing and subjective experience, touching on the hard problem of consciousness."),
    ("Why do we dream?",
     "Consider this: dreams often replay fragments of recent experiences. What might that tell us about the brain's purpose during sleep?",
     "This guides toward memory consolidation theory while encouraging the learner to reason from observation."),
    ("Is math discovered or invented?",
     "Think about the number pi. Did humans create it, or did we find it existing in circles? What does your answer suggest about the nature of mathematics?",
     "The realist vs. nominalist debate in philosophy of mathematics, approached through a concrete example."),
    ("What makes something alive?",
     "A virus can reproduce and evolve but cannot metabolize on its own. Is it alive? What criteria are you using to decide?",
     "Challenges naive definitions of life by presenting a borderline case, encouraging the learner to refine their criteria."),
    ("Can machines truly understand language?",
     "When you read the sentence 'Time flies like an arrow; fruit flies like a banana,' you instantly get the joke. What is it about your understanding that a parser might miss?",
     "Highlights the difference between syntactic parsing and semantic/pragmatic understanding."),
]

EMPATHY_SCENARIOS: List[Tuple[str, str]] = [
    ("I failed my exam and I feel terrible about myself.",
     "I hear you, and I want you to know that one exam doesn't define your intelligence or your worth. It's completely natural to feel disappointed. What matters now is what you do next — would you like to talk about what made this exam particularly hard, so we can figure out a strategy together?"),
    ("I'm so overwhelmed with work I don't know where to start.",
     "That feeling of paralysis when everything seems equally urgent is really tough. Let's take a breath together. One approach that often helps: pick the one task that would give you the most relief if it were done. We can break it into the smallest possible first step. What feels most pressing to you right now?"),
    ("My friend said something hurtful and I don't know what to do.",
     "It's painful when someone we care about says something that stings. Before deciding how to respond, it might help to consider: was this out of character for them? Sometimes people say hurtful things when they're dealing with their own stress. When you feel ready, having an honest conversation about how their words affected you can be really powerful."),
    ("I feel like I'm not good enough compared to everyone else.",
     "That comparison trap is something almost everyone falls into, and it can feel very real. But remember — you're comparing your behind-the-scenes to everyone else's highlight reel. What specific skill or achievement are you proud of, even if it seems small? Let's start there and build on what makes you uniquely you."),
    ("I just lost my pet and I can't stop crying.",
     "I'm so sorry for your loss. The bond with a pet is real and deep, and your grief is completely valid. There's no timeline for grief — let yourself feel what you need to feel. If you'd like, you can tell me about your pet. Sometimes sharing happy memories can bring a small comfort during a hard time."),
    ("I'm scared about the future and feel anxious all the time.",
     "Anxiety about the future is your mind trying to prepare for every possibility — it comes from a place of caring. But living in 'what-ifs' steals the present. One technique that helps: focus on what you can control right now, today. What's one small thing you could do today that your future self would thank you for?"),
]

CREATIVE_RESPONSE_PREFIXES: List[str] = [
    "Here's an interesting way to think about it:",
    "Let me paint a picture for you:",
    "Consider this perspective:",
    "Imagine it this way:",
    "Here's what makes this fascinating:",
    "Think of it like this:",
    "The elegant answer is:",
    "Here's the insight that changes everything:",
    "Let's unpack this step by step:",
    "The key realization is:",
    "A fresh take on this:",
    "What's really happening under the hood:",
    "One way to think about it is:",
    "In addition you could:",
    "Let's test the idea mentally:",
    "The pattern to watch is:",
    "If we strip away jargon:",
    "A clean mental model:",
    "The deeper principle is:",
    "Try viewing it as a tradeoff:",
    "Let's make this concrete:",
    "The intuitive bridge is:",
    "Here's a sharper framing:",
]

REAL_CONVERSATIONS: List[Tuple[str, str]] = [
    ("Hey there, how's it going today?",
     "Hi! I'm doing well, thanks for asking. How can I help you today?"),
    ("I need help with my internet connection, it keeps dropping.",
     "I'm sorry to hear you're experiencing connection issues. Let's troubleshoot this. First, have you tried completely rebooting your router by unplugging it for 30 seconds?"),
    ("Can you recommend a good sci-fi book?",
     "Absolutely! If you enjoy hard sci-fi, 'The Martian' by Andy Weir or 'Three-Body Problem' by Cixin Liu are fantastic. For something more character-driven, 'The Left Hand of Darkness' by Ursula K. Le Guin is a classic. Do you have a preference for space opera or near-future tech?"),
    ("Wow, this weather is absolutely miserable.",
     "It really is! Make sure to stay warm and dry if you have to go out. Is there anything specific you need help with right now, or did you just want a quick chat while escaping the rain?"),
    ("I think I made a mistake on my taxes last year.",
     "That can definitely be stressful, but it's usually fixable. You can typically file an amended return using Form 1040-X. I highly recommend consulting a certified tax professional or CPA to ensure it's handled correctly. Would you like me to find some resources on how the amendment process works?"),
    ("My dog just ate chocolate! What do I do?",
     "This is a serious situation. Chocolate can be very toxic to dogs. Please contact your local emergency veterinarian or the ASPCA Animal Poison Control Center immediately. Have the chocolate wrapper handy so you know the type and amount ingested."),
    ("I'm trying to learn Python but I'm stuck on dictionaries.",
     "Dictionaries can be tricky at first! Think of them like a real dictionary: you have a 'word' (the key) and its 'definition' (the value). For example: `my_dict = {'apple': 'a red fruit'}`. You look up the value by using the key: `print(my_dict['apple'])`. What specific part of using them is tripping you up?"),
    ("I need to return this sweater I bought.",
     "I can help with that. Could you please provide your order number and the email address you used to make the purchase? I'll check our return policy and start the process for you."),
    ("I'm feeling really stressed about work recently.",
     "I hear you. Work stress can really build up and be overwhelming. It's important to acknowledge how you're feeling. Are you dealing with tight deadlines, or is it more about the workload in general? Sometimes taking a short step back or prioritizing tasks can help."),
]

# ---------------------------------------------------------------------------
# User-requested expansions (coverage and creativity)
# ---------------------------------------------------------------------------

ANALOGY_DOMAINS.extend([
    ("Bayes' theorem", "updating your guess after getting a new clue in a mystery"),
    ("HTTP caching", "keeping leftovers in the fridge so you do not cook again"),
    ("database transactions", "making a full bank transfer or canceling it entirely"),
    ("mutex lock", "a single key to a shared bathroom"),
    ("thread scheduling", "a teacher giving each student a turn to speak"),
    ("floating point rounding", "measuring with a ruler that only has coarse markings"),
    ("compression", "zipping a suitcase by packing clothes tightly"),
    ("latency vs throughput", "time for one package to arrive versus packages per hour"),
    ("overfitting", "memorizing last year's exam instead of learning the subject"),
    ("underfitting", "using one rule to solve every kind of puzzle"),
    ("normalization in ML", "rescaling ingredients to the same measuring cup"),
    ("feature engineering", "choosing the best clues before solving a case"),
    ("principal component analysis", "summarizing a messy map with the main roads"),
    ("reinforcement learning", "training a dog with rewards for good behavior"),
    ("backpropagation", "reviewing each step of a recipe to find where it went wrong"),
    ("consensus algorithm", "roommates agreeing on a grocery list before ordering"),
    ("event loop", "a receptionist handling requests one at a time very quickly"),
    ("rate limiting", "a bouncer allowing only so many people in per minute"),
    ("fault tolerance", "a plane with backup systems for critical controls"),
    ("idempotency", "pressing the elevator button twice but sending only one request"),
    ("SQL joins", "matching two guest lists by name"),
    ("vector search", "finding songs by vibe instead of exact title"),
    ("zero-knowledge proof", "proving you know a password without saying it"),
    ("container orchestration", "a dispatcher assigning delivery vans to routes"),
    ("observability", "a car dashboard with gauges, alerts, and service history"),
])

DEBATE_TOPICS.extend([
    "AI-generated art in commercial design",
    "mandatory financial literacy in schools",
    "public transit should be fare-free",
    "drone delivery in cities",
    "geoengineering for climate response",
    "digital IDs issued by governments",
    "banning phones in classrooms",
    "online anonymity protections",
    "sports betting advertising restrictions",
    "water desalination expansion",
    "rent control policies",
    "microplastics regulation",
    "workplace monitoring software",
    "carbon capture investment",
    "open access scientific publishing mandates",
    "AI copilots in healthcare documentation",
    "universal child care",
    "high-speed rail expansion",
    "nuclear fusion funding priorities",
    "biodiversity offsets",
    "robot taxes for automation",
    "mandatory software bill of materials",
    "citizen assemblies for major policies",
    "digital inheritance laws",
    "use of AI in hiring decisions",
])

STORY_GENRES.extend([
    "cli-fi",
    "solarpunk",
    "biopunk",
    "mythic fantasy",
    "urban fantasy",
    "space opera",
    "hard science fiction",
    "magical realism",
    "gothic horror",
    "cosmic horror",
    "folk horror",
    "legal thriller",
    "spy thriller",
    "heist",
    "coming-of-age",
    "literary fiction",
    "slice of life",
    "surrealism",
    "dystopian satire",
    "epistolary",
    "alternate history",
    "western",
    "afrofuturism",
    "dieselpunk",
    "quest fantasy",
])

STORY_STARTERS.extend([
    "The town's clock tower rang thirteen times, and then everyone forgot their names.",
    "By sunrise, every window in the city displayed the same handwritten message.",
    "The expedition returned with one extra crew member nobody recognized.",
    "He opened the toolbox and found a key that hummed when it touched the air.",
    "On the day gravity weakened, the kite shop sold out in minutes.",
    "The courtroom transcript included testimony from someone who had died last year.",
    "At the museum, one painting had changed overnight and now included her face.",
    "The lighthouse beam swept across the valley, even though the sea was miles away.",
    "When the school buried the time capsule, something inside tapped back.",
    "The gardener discovered the roses were blooming in the shape of a map.",
    "Every time the elevator stopped at Floor 9, the passengers arrived younger.",
    "The forecast was perfect, except for one line: 'Do not look up at 4:12 PM.'",
    "She inherited a bookstore where every margin note predicted the next customer.",
    "The village celebrated the comet each century, but this time it changed direction.",
    "His voicemail contained a message timestamped three days in the future.",
    "The subway tunnel ended in a station that wasn't on any city plan.",
    "When the violin string snapped, the storm outside stopped instantly.",
    "At the bottom of the lake, divers found a lit streetlamp.",
    "The recipe book warned, 'Never cook this on a full moon,' so of course they did.",
    "The astronaut's suit radio picked up a child singing from the dark side of the moon.",
    "By dawn, the graffiti on every wall had organized itself into a single poem.",
    "The mayor declared a curfew because shadows had started arriving early.",
    "She pressed play on the cassette labeled 'For when the river speaks.'",
    "The robot bartender remembered a war that had not happened yet.",
    "At the edge of the desert, the road signs pointed to places from his dreams.",
])

MATH_TEMPLATES.extend([
    {"q": "What is the {n}th term of an arithmetic sequence with first term {a1} and common difference {d}?",
     "a": "Use a_n = a1 + (n-1)d. So a_{n} = {a1} + ({n}-1) x {d} = {ans}.",
     "gen": lambda r: (a1 := r.randint(1, 20), d := r.randint(1, 12), n := r.randint(4, 20),
                       {"a1": a1, "d": d, "n": n, "ans": a1 + (n - 1) * d})[-1]},
    {"q": "What is the {n}th term of a geometric sequence with first term {a1} and ratio {ratio}?",
     "a": "Use a_n = a1 x ratio^(n-1). So a_{n} = {a1} x {ratio}^({n}-1) = {ans}.",
     "gen": lambda r: (a1 := r.randint(1, 5), ratio := r.choice([2, 3, 4]), n := r.randint(3, 8),
                       {"a1": a1, "ratio": ratio, "n": n, "ans": a1 * (ratio ** (n - 1))})[-1]},
    {"q": "Find the simple interest on ${p} at {rate}% per year for {years} years.",
     "a": "Simple interest = P x r x t = {p} x {rate}/100 x {years} = ${interest:.2f}.",
     "gen": lambda r: (p := r.randint(100, 5000), rate := r.choice([2, 3, 4, 5, 6, 8, 10]), years := r.randint(1, 8),
                       {"p": p, "rate": rate, "years": years, "interest": p * rate * years / 100})[-1]},
    {"q": "If ${p} grows at {rate}% compounded annually for {years} years, what is the final amount?",
     "a": "Final amount = P(1+r)^t = {p} x (1 + {rate}/100)^{years} = ${amt:.2f}.",
     "gen": lambda r: (p := r.randint(200, 4000), rate := r.choice([3, 4, 5, 6, 8]), years := r.randint(1, 6),
                       {"p": p, "rate": rate, "years": years, "amt": p * ((1 + rate / 100) ** years)})[-1]},
    {"q": "Divide ${total} in the ratio {a}:{b}. How much does each part get?",
     "a": "Total parts = {a}+{b} = {parts}. One part = {total}/{parts} = {unit:.2f}. Shares: ${x:.2f} and ${y:.2f}.",
     "gen": lambda r: (a := r.randint(1, 9), b := r.randint(1, 9), unit := r.randint(5, 40),
                       {"a": a, "b": b, "parts": a + b, "total": unit * (a + b), "unit": float(unit), "x": unit * a, "y": unit * b})[-1]},
    {"q": "Worker A finishes a job in {a_h} hours and Worker B in {b_h} hours. Working together, how long will it take?",
     "a": "Combined rate = 1/{a_h} + 1/{b_h} = {rate:.6f} jobs/hour. Time = 1 / combined rate = {time:.2f} hours.",
     "gen": lambda r: (a_h := r.choice([2, 3, 4, 5, 6, 8, 10, 12]), b_h := r.choice([3, 4, 5, 6, 8, 9, 12]),
                       rate := (1 / a_h + 1 / b_h), {"a_h": a_h, "b_h": b_h, "rate": rate, "time": 1 / rate})[-1]},
    {"q": "Find the distance between the points ({x1}, {y1}) and ({x2}, {y2}).",
     "a": "Distance = sqrt((x2-x1)^2 + (y2-y1)^2) = sqrt(({dx})^2 + ({dy})^2) = sqrt({sq}) = {dist}.",
     "gen": lambda r: (triple := r.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]),
                       x1 := r.randint(-10, 10), y1 := r.randint(-10, 10),
                       sx := r.choice([-1, 1]), sy := r.choice([-1, 1]),
                       x2 := x1 + sx * triple[0], y2 := y1 + sy * triple[1],
                       {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "dx": x2 - x1, "dy": y2 - y1, "sq": triple[0] ** 2 + triple[1] ** 2, "dist": triple[2]})[-1]},
    {"q": "A circle has radius {r}. What are its circumference and area? (Use pi = 3.14)",
     "a": "Circumference = 2pi r = 2 x 3.14 x {r} = {circ:.2f}. Area = pi r^2 = 3.14 x {r}^2 = {area:.2f}.",
     "gen": lambda r: (rad := r.randint(1, 25), {"r": rad, "circ": 2 * 3.14 * rad, "area": 3.14 * rad * rad})[-1]},
    {"q": "A right triangle has legs {a} and {b}. What is the hypotenuse?",
     "a": "Use the Pythagorean theorem: c = sqrt({a}^2 + {b}^2) = sqrt({sq}) = {c}.",
     "gen": lambda r: (triple := r.choice([(3, 4, 5), (5, 12, 13), (8, 15, 17), (9, 12, 15)]),
                       {"a": triple[0], "b": triple[1], "sq": triple[0] ** 2 + triple[1] ** 2, "c": triple[2]})[-1]},
    {"q": "A quantity increases from {old} to {new}. What is the percent increase?",
     "a": "Percent increase = ((new-old)/old) x 100 = (({new}-{old})/{old}) x 100 = {pct:.2f}%.",
     "gen": lambda r: (old := r.randint(20, 200), inc := r.randint(1, 150), {"old": old, "new": old + inc, "pct": inc * 100 / old})[-1]},
    {"q": "A price drops from ${old} to ${new}. What is the percent decrease?",
     "a": "Percent decrease = ((old-new)/old) x 100 = (({old}-{new})/{old}) x 100 = {pct:.2f}%.",
     "gen": lambda r: (old := r.randint(30, 400), pct_base := r.choice([5, 10, 15, 20, 25, 30]),
                       new := old - round(old * pct_base / 100), {"old": old, "new": new, "pct": (old - new) * 100 / old})[-1]},
    {"q": "If {qty} items cost ${total:.2f}, what is the unit price per item?",
     "a": "Unit price = total / quantity = ${total:.2f} / {qty} = ${unit:.2f} per item.",
     "gen": lambda r: (qty := r.randint(2, 25), unit := r.choice([0.5, 0.75, 1.25, 2.5, 3.2, 4.8, 6.4]),
                       {"qty": qty, "total": qty * unit, "unit": unit})[-1]},
    {"q": "Find the mean of these numbers: {nums}.",
     "a": "Add the numbers: {sumv}. There are {n} numbers. Mean = {sumv}/{n} = {mean:.2f}.",
     "gen": lambda r: (vals := [r.randint(1, 30) for _ in range(r.randint(4, 7))],
                       {"nums": ', '.join(map(str, vals)), "sumv": sum(vals), "n": len(vals), "mean": sum(vals) / len(vals)})[-1]},
    {"q": "A course grade is {hw}% homework and {exam}% exam. If homework is {hw_score}% and exam is {exam_score}%, what is the weighted average?",
     "a": "Weighted grade = ({hw}/100) x {hw_score} + ({exam}/100) x {exam_score} = {final:.2f}%.",
     "gen": lambda r: (hw := r.choice([30, 40, 50, 60]), exam := 100 - hw, hw_score := r.randint(60, 100), exam_score := r.randint(50, 100),
                       {"hw": hw, "exam": exam, "hw_score": hw_score, "exam_score": exam_score, "final": hw * hw_score / 100 + exam * exam_score / 100})[-1]},
    {"q": "You mix {v1} L of a {p1}% solution with {v2} L of a {p2}% solution. What is the final concentration?",
     "a": "Solute = {v1}x{p1}% + {v2}x{p2}% = {solute:.2f} L-equivalent. Total volume = {tot} L. Concentration = {solute:.2f}/{tot} = {pct:.2f}%.",
     "gen": lambda r: (v1 := r.randint(1, 10), v2 := r.randint(1, 10), p1 := r.choice([5, 10, 15, 20, 25, 30]), p2 := r.choice([35, 40, 50, 60, 70]),
                       solute := v1 * p1 / 100 + v2 * p2 / 100, {"v1": v1, "v2": v2, "p1": p1, "p2": p2, "solute": solute, "tot": v1 + v2, "pct": 100 * solute / (v1 + v2)})[-1]},
    {"q": "Convert {ms} m/s to km/h.",
     "a": "1 m/s = 3.6 km/h. So {ms} x 3.6 = {kmh:.2f} km/h.",
     "gen": lambda r: (ms := r.randint(1, 40), {"ms": ms, "kmh": ms * 3.6})[-1]},
    {"q": "Two cars start {dist} km apart and drive toward each other at {v1} km/h and {v2} km/h. When do they meet?",
     "a": "Closing speed = {v1} + {v2} = {vsum} km/h. Time = distance / closing speed = {dist}/{vsum} = {time:.2f} hours.",
     "gen": lambda r: (v1 := r.randint(30, 90), v2 := r.randint(30, 90), t := r.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
                       dist := int((v1 + v2) * t), {"dist": dist, "v1": v1, "v2": v2, "vsum": v1 + v2, "time": dist / (v1 + v2)})[-1]},
    {"q": "A bag has {reds} red and {blues} blue marbles. What is the probability of drawing a red marble?",
     "a": "Probability = favorable / total = {reds}/({reds}+{blues}) = {reds}/{tot} = {prob:.4f}.",
     "gen": lambda r: (reds := r.randint(1, 12), blues := r.randint(1, 12), {"reds": reds, "blues": blues, "tot": reds + blues, "prob": reds / (reds + blues)})[-1]},
    {"q": "How many different ways can {n} distinct books be arranged on a shelf?",
     "a": "Arrangements of {n} distinct items = {n}! = {ans}.",
     "gen": lambda r: (n := r.randint(3, 8), {"n": n, "ans": math.factorial(n)})[-1]},
    {"q": "How many combinations are there to choose {k} students from {n} students?",
     "a": "Use combinations: C(n,k) = n! / (k!(n-k)!) = C({n},{k}) = {ans}.",
     "gen": lambda r: (n := r.randint(5, 12), k := r.randint(2, 4), k := min(k, n - 1), {"n": n, "k": k, "ans": math.comb(n, k)})[-1]},
    {"q": "Solve the system: x + y = {s} and x - y = {d}.",
     "a": "Add the equations: 2x = {s} + {d} = {twox}, so x = {x}. Then y = {s} - {x} = {y}.",
     "gen": lambda r: (x := r.randint(-10, 20), y := r.randint(-10, 20), {"s": x + y, "d": x - y, "twox": 2 * x, "x": x, "y": y})[-1]},
    {"q": "Solve x^2 - {sumv}x + {prod} = 0.",
     "a": "Find two numbers that add to {sumv} and multiply to {prod}: {r1} and {r2}. So (x-{r1})(x-{r2})=0, giving x = {r1} or x = {r2}.",
     "gen": lambda r: (r1 := r.randint(1, 12), r2 := r.randint(1, 12), {"sumv": r1 + r2, "prod": r1 * r2, "r1": r1, "r2": r2})[-1]},
    {"q": "A population starts at {start} and doubles every {period} hours. What is it after {n} doubling periods?",
     "a": "After n doublings: final = start x 2^n = {start} x 2^{n} = {final}. Total elapsed time is {hours} hours.",
     "gen": lambda r: (start := r.randint(10, 500), period := r.choice([1, 2, 3, 4]), n := r.randint(2, 8),
                       {"start": start, "period": period, "n": n, "final": start * (2 ** n), "hours": period * n})[-1]},
    {"q": "An asset worth ${start} depreciates by {rate}% each year. What is it worth after {years} years?",
     "a": "Value after depreciation = start x (1-rate/100)^years = {start} x (1-{rate}/100)^{years} = ${final:.2f}.",
     "gen": lambda r: (start := r.randint(1000, 20000), rate := r.choice([5, 8, 10, 12, 15, 20]), years := r.randint(1, 6),
                       {"start": start, "rate": rate, "years": years, "final": start * ((1 - rate / 100) ** years)})[-1]},
    {"q": "A rectangle's length and width are each scaled by a factor of {k}. By what percent does the area change?",
     "a": "Area scales by k^2. Here, k = {k}, so area factor = {factor:.2f}. Percent change = ({factor:.2f} - 1) x 100 = {pct:.2f}%.",
     "gen": lambda r: (k := r.choice([0.5, 0.8, 1.5, 2.0, 2.5, 3.0]), factor := k * k, {"k": k, "factor": factor, "pct": (factor - 1) * 100})[-1]},
])

LOGIC_PUZZLES.extend([
    ("A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
     "The ball costs $0.05. Let the ball be x and the bat be x + 1. Then x + (x + 1) = 1.10, so 2x = 0.10 and x = 0.05."),
    ("You pass the person in second place in a race. What place are you in now?",
     "Second place. You take the position of the runner you passed."),
    ("A lily pad patch doubles in size every day and covers the lake in 30 days. On what day was it half the lake?",
     "Day 29. If it doubles each day, it must have been half the size the day before it covered the whole lake."),
    ("There are 3 switches downstairs and one light bulb upstairs. You may go upstairs once. How can you identify the correct switch?",
     "Turn on switch 1 for a few minutes, then switch it off and turn on switch 2. Go upstairs: if the bulb is on, it is switch 2; if off but warm, switch 1; if off and cold, switch 3."),
    ("A clock shows 3:15. What is the smaller angle between the hour and minute hands?",
     "7.5 degrees. The minute hand is at 90 degrees. The hour hand is at 3.25 x 30 = 97.5 degrees. Difference = 7.5 degrees."),
    ("If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
     "Yes. This is a valid transitive syllogism."),
    ("What is the next letter in the sequence J, F, M, A, M, J, J, A, S, O, N, ...?",
     "D. These are the first letters of the months January through November; next is December."),
    ("You have 8 identical-looking coins, but one is heavier. What is the minimum number of balance-scale weighings needed to guarantee finding it?",
     "Two weighings. Weigh 3 vs 3. If equal, the heavy coin is in the remaining 2 and one weighing finds it. If not equal, it is in the heavier group of 3 and one more weighing identifies it."),
    ("A man lives on the 20th floor, takes the elevator down each morning, but usually rides only to the 10th floor on the way up and walks the rest. Why?",
     "He is short and can reach only the 10th-floor button. On rainy days he uses an umbrella to reach the 20th-floor button."),
    ("If yesterday was two days before Friday, what day is today?",
     "Thursday. Two days before Friday is Wednesday, so if yesterday was Wednesday, today is Thursday."),
    ("A family has two children. At least one is a boy. Is it more likely they have two boys or one boy and one girl?",
     "One boy and one girl. Given at least one boy, the equally likely cases are BB, BG, GB. Two boys is 1 out of 3; mixed is 2 out of 3."),
    ("What comes next: 1, 4, 9, 16, 25, ...?",
     "36. These are perfect squares: 1^2, 2^2, 3^2, 4^2, 5^2, so the next is 6^2."),
    ("A cube is painted on all sides and cut into 27 equal cubes. How many small cubes have paint on exactly two faces?",
     "12. These are the edge-center cubes, one on each edge of the large cube."),
    ("You have a 5-liter jug and a 3-liter jug. How can you measure exactly 4 liters?",
     "Fill the 5-liter jug and pour into the 3-liter jug, leaving 2 liters. Empty the 3-liter jug, pour in the 2 liters, fill the 5-liter jug again, and pour into the 3-liter jug until full (1 liter). Four liters remain in the 5-liter jug."),
    ("If a statement is false, then the moon is made of cheese. The statement is false. What can you conclude about the moon?",
     "Nothing. A conditional with a false antecedent can be true regardless of the conclusion."),
    ("A number is between 1 and 100. It is greater than 50 and not divisible by 3. What is the maximum possible number?",
     "100. It is greater than 50 and not divisible by 3."),
    ("A sequence goes 10, 7, 8, 5, 6, 3, ... What is next?",
     "4. The pattern alternates -3, +1."),
    ("How many handshakes occur if 6 people each shake hands with every other person exactly once?",
     "15 handshakes. The count is C(6,2) = 6 x 5 / 2 = 15."),
    ("A father is 30 years older than his son. In 5 years, he will be 3 times the son's age. How old is the son now?",
     "10 years old. Let the son's age be s. Then s + 35 = 3(s + 5), so 20 = 2s and s = 10."),
    ("What is the odd one out: triangle, square, circle, cube?",
     "Cube, because it is a 3D solid while the others are 2D shapes."),
    ("A 4-digit password uses digits 0-9 and digits can repeat. How many possible passwords are there?",
     "10,000. Each of the 4 positions has 10 choices, so 10^4 = 10,000."),
    ("You flip two fair coins. Which is more likely: exactly one head or two heads?",
     "Exactly one head. Outcomes are HH, HT, TH, TT, so exactly one head occurs in 2 out of 4 outcomes while two heads occurs in 1 out of 4."),
    ("Which weighs more: a pound of feathers or a pound of bricks?",
     "They weigh the same: one pound each."),
    ("A train leaves City A at 60 mph toward City B. Another leaves City B at 40 mph toward City A. Which train is closer to City B when they meet?",
     "Neither. When they meet, they are at the same location."),
    ("Five people can build five chairs in five hours at the same rate. How long would ten people take to build ten chairs?",
     "Five hours. Each person builds one chair in five hours."),
])

SOCRATIC_SEEDS.extend([
    ("Do humans have free will?",
     "When you make a choice, how much of it feels authored by you versus shaped by habits, environment, and biology? What evidence would change your view?",
     "This invites examination of agency while grounding the debate in experience and testable assumptions."),
    ("What is justice?",
     "If two people break the same rule under very different circumstances, should justice treat them identically or consider context?",
     "This distinguishes equality, equity, and context-sensitive fairness."),
    ("What does it mean to know something?",
     "If you strongly believe a claim and it turns out true by luck, did you know it or merely guess correctly?",
     "This points toward justification and the limits of certainty."),
    ("Are you the same person you were ten years ago?",
     "If your memories, values, and body have changed, what exactly provides continuity of identity?",
     "This probes personal identity through change over time."),
    ("What makes art meaningful?",
     "Is meaning located in the artist's intention, the audience's interpretation, or features of the work itself?",
     "This opens interpretive frameworks without assuming a single authority."),
    ("Can a law be legal but immoral?",
     "If a law is validly passed yet harms a vulnerable group, what should citizens prioritize: obedience or conscience?",
     "This separates legality from ethics and introduces civil disobedience reasoning."),
    ("Is equality always fair?",
     "If two runners start at different distances from the finish line, is giving them the same shoes enough to make the race fair?",
     "This guides toward the difference between equal treatment and equitable support."),
    ("What is courage?",
     "Is courage the absence of fear, or acting well despite fear? Which examples support your answer?",
     "This reframes courage as disciplined action rather than emotionlessness."),
    ("Can science answer every important question?",
     "Which questions are scientific, and which require ethical, philosophical, or artistic reasoning?",
     "This clarifies the scope and limits of empirical methods."),
    ("Do words shape thought?",
     "If a language has many words for a concept, does that sharpen perception of it or simply label distinctions people already notice?",
     "This tests linguistic relativity without overstating it."),
    ("What is intelligence?",
     "If someone struggles on exams but excels at building systems or reading social dynamics, are they less intelligent or differently intelligent?",
     "This challenges narrow measurements of intelligence."),
    ("What makes a friendship strong?",
     "Is a good friend the one who always agrees, or the one who can disagree honestly while preserving trust?",
     "This emphasizes trust and repair over simple harmony."),
    ("Can uncertainty be useful?",
     "How might lower confidence in your answer make your reasoning better?",
     "This points toward intellectual humility as a strength."),
    ("What gives life meaning?",
     "Do people discover meaning, create it, or both? What in your life supports your view?",
     "This keeps the discussion grounded while exploring existential frameworks."),
    ("Is privacy still possible in a digital world?",
     "If convenience requires sharing data, where would you draw the line and why?",
     "This frames privacy as a tradeoff rather than an all-or-nothing category."),
    ("When does progress become harmful?",
     "If a technology improves efficiency but weakens autonomy or community, how should we evaluate that tradeoff?",
     "This introduces multi-dimensional evaluation of innovation."),
    ("Do animals have rights?",
     "What capacities should matter when deciding who deserves moral consideration?",
     "This examines moral status criteria such as sentience and vulnerability."),
    ("Can AI be creative?",
     "If a system produces original-seeming work by recombining patterns, how is that similar to and different from human creativity?",
     "This pushes on originality, authorship, and process."),
    ("What is a good explanation?",
     "Would you prefer an exact explanation that is hard to understand or a simple one that is slightly incomplete? Why?",
     "This surfaces tradeoffs between precision and pedagogy."),
    ("Is memory reliable?",
     "If two honest people remember the same event differently, what should that teach us about confidence in recollection?",
     "This opens reconstruction theory and witness reliability."),
    ("What is responsibility in a complex system?",
     "When harm results from many small decisions, how should responsibility be assigned?",
     "This is useful for ethics in organizations and technology systems."),
    ("Can someone be rational and wrong?",
     "If a person reasons carefully from incomplete information, is the problem irrationality or a data limitation?",
     "This distinguishes reasoning quality from factual correctness."),
    ("What is the purpose of education?",
     "Is education mainly for jobs, citizenship, personal growth, or something else? What follows from prioritizing one goal?",
     "This clarifies hidden assumptions in education debates."),
    ("How do we decide what counts as evidence?",
     "When should personal experience matter, and when should broader data outweigh it?",
     "This examines evidence hierarchies and context."),
    ("What makes a question good?",
     "Which kinds of questions produce shallow answers, and how could you reframe one to invite better reasoning?",
     "This develops meta-reasoning about inquiry quality."),
])

EMPATHY_SCENARIOS.extend([
    ("I got rejected from a job I really wanted and now I feel embarrassed.",
     "That hurts, especially when you let yourself hope for it. Rejection is not a clean measure of your value. If you want, we can turn this into a next-step plan and use it instead of letting it become a verdict."),
    ("I keep procrastinating and then hating myself for it.",
     "That cycle is exhausting because the task and the self-judgment stack together. You do not need to solve everything today. Let's reduce friction and start with a five-minute version of the task."),
    ("I made a mistake at work and I can't stop replaying it.",
     "That replay loop usually means you care and want to improve. If the mistake can be repaired, a concrete repair plan often reduces the spiral. We can outline the fix and the message to send."),
    ("My partner and I keep having the same argument.",
     "Repeated arguments often point to an underlying need that does not feel heard. It can help to pause the debate and ask what each person is trying to protect. I can help you phrase that conversation."),
    ("I'm lonely even when I'm around people.",
     "That kind of loneliness is real and can feel confusing. It usually points to missing connection, not missing company. What kinds of conversations leave you feeling more seen?"),
    ("I feel guilty for taking a day off.",
     "Rest is not a moral failure. Recovery is part of doing good work over time. Taking a day off can be a responsible decision, not a selfish one."),
    ("I'm behind in life compared to my friends.",
     "Comparing timelines can make normal variation feel like failure. Lives are not synchronized projects. It may help to define progress by your values instead of other people's milestones."),
    ("I have to give a presentation and I'm panicking.",
     "That panic is common and does not mean you will perform badly. Your body is preparing for stress, not predicting the outcome. We can lower the fear with a short opening script and a few anchor points."),
    ("I feel numb and I don't know why.",
     "Feeling numb can happen after sustained stress. It can be a protective response, even if it feels unsettling. We can start by naming what your last few days have looked like without forcing a perfect explanation."),
    ("My family doesn't understand the career path I chose.",
     "That can feel like being unseen, not just disagreed with. You are allowed to want both approval and autonomy. We can work on a clear explanation that protects your boundaries."),
    ("I keep comparing my body to people online.",
     "That comparison loop can be brutal because online images are curated and often edited. The emotional impact is real. It may help to shift the focus toward how you want to feel and function, not just how you look."),
    ("I snapped at someone and now I feel ashamed.",
     "Shame can make repair harder, but repair is still possible. A sincere apology that names the impact and avoids excuses can go a long way. I can help you draft one."),
    ("I can't focus and everything feels mentally noisy.",
     "Mental noise can make simple tasks feel heavy. Instead of pushing harder, try reducing inputs for a short block. What is one task we can protect for 10 minutes with fewer tabs and notifications?"),
    ("I'm worried I chose the wrong major.",
     "That uncertainty is common and it does not mean you failed. A major matters, but it is not a permanent verdict. We can look at which parts drain you and which kinds of problems energize you."),
    ("I miss someone I know I shouldn't go back to.",
     "Missing them does not automatically mean the relationship was good for you. It often means the bond was significant. You can honor the feeling without reopening something that hurt you."),
    ("I feel like I let my team down.",
     "That is hard because it touches both responsibility and belonging. Trust can often be rebuilt with clear ownership and consistent follow-through. What is the first repair step you can take?"),
    ("I'm exhausted from taking care of everyone else.",
     "Caregiving fatigue is real, and it can build quietly. Needing support does not cancel your compassion. What is one responsibility you could lighten this week, even a little?"),
    ("I keep overthinking texts before I send them.",
     "Overthinking often comes from trying to avoid misunderstanding or rejection. A good rule is clear, kind, and specific. If you want, paste the draft and we can simplify it."),
    ("I failed again and I'm starting to think I just can't do this.",
     "Repeated setbacks can make 'not yet' feel like 'never.' That feeling is understandable, but it is not proof. Let's look at what changed between attempts so we can find leverage instead of just blame."),
    ("I feel bad saying no to people.",
     "Saying no can feel like risking connection, especially if you are used to being dependable. Boundaries often protect relationships from resentment. We can make your no both kind and firm."),
    ("I have a big test tomorrow and my mind goes blank when I study.",
     "Stress can block recall even when you know the material. That does not mean it is gone. Short review cycles, retrieval practice, and sleep usually help more than panic-cramming."),
    ("I am scared to start because what if I am not good at it?",
     "Starting can feel risky because it creates evidence. But staying still can quietly strengthen the same fear. A tiny start gives you information without demanding perfection."),
    ("My friend group is changing and I feel left behind.",
     "Friendship transitions can feel like grief, even when no one is at fault. It is okay to mourn what changed. We can think about how to preserve what matters and build new connection too."),
    ("I can't stop thinking about an awkward thing I said.",
     "Your brain is probably spotlighting it more than anyone else is. That does not erase the discomfort, but it can shrink its size. If needed, a short follow-up message can help more than hours of rumination."),
    ("I am burned out but I cannot take a break right now.",
     "When a full break is not possible, micro-recovery matters. The goal becomes reducing harm and preserving function, not pretending everything is fine. We can make a short survival plan for this week."),
    ("I am trying to be healthier but I keep falling off track.",
     "Setbacks are part of behavior change, not proof you cannot change. The key question is often whether the plan was sustainable. What habit feels easiest to restart today?"),
    ("I feel invisible in meetings.",
     "That can wear you down over time. Sometimes a small strategy shift helps, like speaking earlier or preparing one key point. We can script a strong opening line for your next meeting."),
    ("I regret wasting so much time in the past.",
     "Regret is painful, but it can also clarify what matters now. You cannot recover old time, but you can use what you learned to shape the next season. What do you want to stop postponing?"),
    ("I am nervous about moving to a new city where I know nobody.",
     "That mix of excitement and fear is normal. New places often feel lonely before they feel full of possibility. We can build a first-month plan for routines and connection."),
    ("I feel like people only like me when I'm useful.",
     "That feeling can create a lot of pressure to keep performing. You deserve relationships where you are valued, not just needed. It may help to notice who checks on you when you have nothing to offer."),
    ("I am angry all the time lately and I don't like who I'm becoming.",
     "Anger often sits on top of pain, exhaustion, or crossed boundaries. Noticing it and wanting to respond differently is already important progress. We can look at the patterns and triggers together."),
    ("I am struggling with a chronic problem and people keep telling me to stay positive.",
     "That can feel dismissive when you are carrying something real every day. You do not owe anyone constant positivity. It is okay to want support that feels honest and practical."),
    ("I feel guilty for being sad because other people have it worse.",
     "Pain does not need to win a competition to matter. Someone else's suffering does not erase yours. You are allowed to take your feelings seriously and still care deeply about others."),
    ("I am afraid to ask for help because I don't want to be a burden.",
     "A lot of people feel this, especially those used to helping others. Asking for support can be an act of trust, not a burden. People often prefer to know you need help rather than watch you struggle alone."),
    ("I cannot tell if I should keep pushing through or quit this project.",
     "That is a difficult decision, and it is not just grit versus failure. A better question is whether the project still fits your goal and whether the current cost is worth it. We can evaluate it without judging you."),
])

CREATIVE_RESPONSE_PREFIXES.extend([
    "A counterintuitive angle:",
    "If we zoom out for a second:",
    "Let's build intuition first:",
    "Picture a small experiment:",
    "The practical version is:",
    "From a systems point of view:",
    "In plain English:",
    "The short answer with nuance:",
    "A metaphor that helps:",
    "Here's a creative twist:",
    "Let's reason from first principles:",
    "The surprising part is:",
    "A storyteller's version:",
    "A scientist's version:",
    "A builder's version:",
    "What this reminds me of:",
    "Let's test the idea mentally:",
    "The pattern to watch is:",
    "If we strip away jargon:",
    "A clean mental model:",
    "The deeper principle is:",
    "Try viewing it as a tradeoff:",
    "Let's make this concrete:",
    "The intuitive bridge is:",
    "Here's a sharper framing:",
])

# ---------------------------------------------------------------------------
# Large combinatorial seed expansions (+1000 each requested category, plus
# optional follow-on increments)
# ---------------------------------------------------------------------------

def _append_exact_count(target: List[Any], generated: List[Any], count: int, label: str) -> None:
    if len(generated) < count:
        raise ValueError(f"{label}: generated {len(generated)} < requested {count}")
    target.extend(generated[:count])


def _generate_bulk_analogy_domains(count: int = 1000) -> List[Tuple[str, str]]:
    mechanics = [
        "feedback loops", "signal routing", "error correction", "resource allocation",
        "synchronization", "progressive refinement", "priority scheduling", "load shedding",
        "state transitions", "distributed coordination",
    ]
    systems = [
        "robotics", "cloud systems", "traffic networks", "supply chains", "classrooms",
        "hospitals", "ecosystems", "financial markets", "game design", "research labs",
    ]
    goals = [
        "under uncertainty", "at scale", "under tight constraints", "with limited memory",
        "during peak demand", "with noisy inputs", "across long distances",
        "with competing goals", "under real-time pressure", "over time",
    ]
    analogy_places = [
        "a busy kitchen", "an airport control tower", "a city bus terminal",
        "a theater backstage", "a newsroom on deadline", "a warehouse at holiday rush",
        "a hospital triage desk", "a symphony rehearsal", "a shipping port", "a pit crew",
    ]
    analogy_actions = [
        "workers coordinate tasks by passing updates",
        "small delays cascade unless someone reprioritizes",
        "teams split work to avoid bottlenecks",
        "signals must arrive in the right order",
        "everyone shares a limited set of tools",
        "mistakes are caught and corrected before shipping",
        "new arrivals are sorted by urgency",
        "backup plans take over when something fails",
        "specialists focus on different parts of the same problem",
        "the plan is revised as new information arrives",
    ]
    analogy_constraints = [
        "time matters more than perfection",
        "resources are limited",
        "communication can be noisy",
        "some tasks depend on earlier steps",
        "unexpected disruptions happen",
        "multiple goals compete",
        "not every worker sees the whole system",
        "priorities keep changing",
        "quality checks still have to happen",
        "the environment keeps shifting",
    ]
    out: List[Tuple[str, str]] = []
    for mech in mechanics:
        for system in systems:
            for goal in goals:
                place = analogy_places[len(out) % len(analogy_places)]
                action = analogy_actions[(len(out) // 3) % len(analogy_actions)]
                constraint = analogy_constraints[(len(out) // 7) % len(analogy_constraints)]
                concept = f"{mech} in {system} {goal}"
                analogy = f"{place} where {action} while {constraint}"
                out.append((concept, analogy))
                if len(out) >= count:
                    return out
    return out


def _generate_bulk_debate_topics(count: int = 1000) -> List[str]:
    domains = [
        "AI safety", "public health", "urban planning", "education policy",
        "energy infrastructure", "water management", "workplace automation",
        "digital privacy", "scientific funding", "transportation systems",
    ]
    actions = [
        "should prioritize local control over national standards",
        "should be regulated through licensing frameworks",
        "should receive tax incentives",
        "should face stricter transparency requirements",
        "should be expanded through public investment",
        "should rely more on market-based solutions",
        "should include stronger consumer protections",
        "should be limited until long-term effects are clearer",
        "should be taught earlier in schools",
        "should use independent oversight boards",
    ]
    scopes = [
        "in low-income communities", "in rural regions", "in major cities",
        "for small businesses", "for public institutions", "for private companies",
        "during emergencies", "for long-term resilience", "for youth populations",
        "for international coordination",
    ]
    out: List[str] = []
    for domain in domains:
        for action in actions:
            for scope in scopes:
                out.append(f"whether {domain} {action} {scope}")
                if len(out) >= count:
                    return out
    return out


def _generate_bulk_story_genres(count: int = 1000) -> List[str]:
    moods = ["hopeful", "bleak", "wry", "lyrical", "tense", "melancholic", "playful", "epic", "intimate", "haunting"]
    settings = ["urban", "cosmic", "frontier", "subterranean", "coastal", "desert", "forest", "arctic", "virtual", "post-industrial"]
    structures = ["mystery", "romance", "quest", "heist", "survival tale", "political drama", "family saga", "revenge tale", "coming-of-age", "courtroom drama"]
    out: List[str] = []
    for mood in moods:
        for setting in settings:
            for structure in structures:
                out.append(f"A {mood} {structure} set in a {setting} environment.")
                if len(out) >= count:
                    return out
    return out


def _generate_bulk_real_conversations(count: int = 1000) -> List[Tuple[str, str]]:
    intents = [
        "Inquiring about", "Complaining about", "Seeking advice on", "Sharing an opinion on",
        "Asking for a recommendation regarding", "Reporting an issue with",
        "Expressing gratitude for", "Ranting about", "Confused about", "Excited about",
        "Asking for help with", "Following up on", "Suggesting an improvement for",
        "Venting frustration over", "Celebrating a success involving"
    ]
    topics = [
        "a delayed flight", "a confusing software update", "a tricky recipe", 
        "managing personal finances", "a recent movie", "a strange noise in the car",
        "a difficult conversation with a friend", "planning a vacation", "a home repair",
        "learning a new language", "a frustrating customer service experience",
        "buying a new laptop", "a lost package", "an upcoming job interview",
        "training a new puppy"
    ]
    tones = [
        "sympathetic and helpful", "professional and direct", "casual and friendly",
        "empathetic and supportive", "informative and structured",
        "patient and encouraging", "enthusiastic and upbeat", "calm and reassuring"
    ]
    out: List[Tuple[str, str]] = []
    for intent in intents:
        for topic in topics:
            for tone in tones:
                prompt = f"User is {intent.lower()} {topic}."
                response = f"[Bot responding in a {tone} manner, addressing the user's situation directly and offering relevant engagement or assistance regarding {topic}.]"
                out.append((prompt, response))
                if len(out) >= count:
                    return out
    return out


def _generate_bulk_story_starters(count: int = 1000) -> List[str]:
    subjects = [
        "the town siren", "the oldest bridge", "a sealed observatory", "the mayor's radio",
        "a forgotten subway line", "the school greenhouse", "the harbor lighthouse",
        "a museum basement", "the last weather station", "a roadside diner sign",
    ]
    events = [
        "began speaking in a voice no one recognized",
        "lit up despite the blackout",
        "appeared on maps that did not exist yesterday",
        "started counting backward at midnight",
        "opened only during thunderstorms",
        "broadcast the same warning every hour",
        "vanished and returned in a different place",
        "showed a date from the future",
        "drew strangers to it from miles away",
        "reacted when she touched it",
    ]
    consequences = [
        "the police blocked the street before sunrise.",
        "everyone in town claimed they had dreamed it first.",
        "he finally understood why his grandmother never spoke about the war.",
        "the river changed course by morning.",
        "the newspaper printed an extra edition at dawn.",
        "three people disappeared trying to investigate.",
        "the power grid failed in every surrounding county.",
        "the celebration turned into an evacuation within an hour.",
        "the message on it matched the one in her notebook.",
        "nobody could agree whether it had always been there.",
    ]
    out: List[str] = []
    for subject in subjects:
        for event in events:
            for consequence in consequences:
                out.append(f"When {subject} {event}, {consequence}")
                if len(out) >= count:
                    return out
    return out


def _generate_bulk_creative_prefixes(count: int = 1000) -> List[str]:
    openers = [
        "From a", "Using a", "Take a", "Try a", "Here is a", "Let's use a",
        "Consider a", "Start with a", "Build a", "Shift to a",
    ]
    lenses = [
        "systems", "first-principles", "storytelling", "engineering", "teacher",
        "scientist", "designer", "strategist", "builder", "debugger",
    ]
    styles = [
        "lens", "frame", "mental model", "walkthrough", "thought experiment",
        "comparison", "breakdown", "viewpoint", "interpretation", "angle",
    ]
    closers = [
        "for intuition", "for clarity", "for decision-making", "for beginners",
        "for practical use", "for deeper understanding", "for sharp reasoning",
        "for fast learning", "for conversation", "for problem-solving",
    ]
    out: List[str] = []
    for opener in openers:
        for lens in lenses:
            for style in styles:
                closer = closers[len(out) % len(closers)]
                out.append(f"{opener} {lens} {style} {closer}:")
                if len(out) >= count:
                    return out
    return out


def _generate_bulk_socratic_seeds(count: int = 1000) -> List[Tuple[str, str, str]]:
    themes = [
        "truth", "fairness", "freedom", "identity", "knowledge", "evidence",
        "progress", "responsibility", "creativity", "intelligence",
        "cooperation", "conflict", "trust", "power", "privacy",
        "language", "meaning", "risk", "beauty", "wisdom",
    ]
    contexts = [
        "in science", "in politics", "in friendship", "in education", "at work",
        "in technology", "in art", "in law", "in family life", "online",
    ]
    prompts = [
        "What would count as a strong example of this?",
        "Which assumption are you relying on most?",
        "What case would challenge your current view?",
        "How would you explain your answer to someone who disagrees?",
        "What tradeoff might you be overlooking?",
    ]
    out: List[Tuple[str, str, str]] = []
    for theme in themes:
        for context in contexts:
            for prompt in prompts:
                question = f"What does {theme} really mean {context}?"
                response = (
                    f"Before answering, try defining {theme} in your own words {context}. "
                    f"{prompt} Then ask whether your definition still works in an edge case."
                )
                insight = (
                    f"This Socratic seed trains conceptual precision about {theme} {context} "
                    f"and encourages the learner to test assumptions against counterexamples."
                )
                out.append((question, response, insight))
                if len(out) >= count:
                    return out
    return out


def _generate_bulk_empathy_scenarios(count: int = 1000) -> List[Tuple[str, str]]:
    feelings = [
        "anxious", "discouraged", "embarrassed", "overwhelmed", "frustrated",
        "stuck", "lonely", "ashamed", "uncertain", "drained",
    ]
    contexts = [
        "a job search", "school deadlines", "a difficult conversation", "a team project",
        "starting a new habit", "family expectations", "financial stress", "social situations",
        "a creative project", "learning a hard skill", "health routines", "public speaking",
        "career decisions", "moving to a new place", "friendship changes", "online comparison",
        "work mistakes", "burnout", "setting boundaries", "long-term goals",
    ]
    blockers = [
        "I keep thinking one mistake means I am failing",
        "I do not know what to do first",
        "I am afraid people will judge me",
        "I feel behind everyone else",
        "my motivation disappears when I get stressed",
    ]
    out: List[Tuple[str, str]] = []
    for feeling in feelings:
        for context in contexts:
            for blocker in blockers:
                scenario = f"I feel {feeling} about {context} because {blocker}."
                response = (
                    f"That sounds really hard, and it makes sense to feel {feeling} in that situation. "
                    f"The fact that you are noticing this means you care. "
                    f"Instead of solving everything at once, try one small next step you can finish today. "
                    f"If you want, we can break {context} into something more manageable together."
                )
                out.append((scenario, response))
                if len(out) >= count:
                    return out
    return out


def _gen_bulk_linear(r: random.Random) -> Dict[str, Any]:
    a = r.randint(2, 12)
    x = r.randint(-12, 24)
    b = r.randint(-20, 30)
    c = a * x + b
    return {"a": a, "b": b, "c": c, "rhs": c - b, "x": float(x)}


def _gen_bulk_discount(r: random.Random) -> Dict[str, Any]:
    p = r.choice([5, 10, 15, 20, 25, 30, 35, 40, 50])
    price = r.randint(20, 600)
    disc = price * p / 100
    return {"p": p, "price": price, "disc": disc, "final": price - disc}


def _gen_bulk_rect(r: random.Random) -> Dict[str, Any]:
    l = r.randint(2, 60)
    w = r.randint(2, 60)
    return {"l": l, "w": w, "area": l * w, "perim": 2 * (l + w)}


def _gen_bulk_avg_speed(r: random.Random) -> Dict[str, Any]:
    t1 = r.randint(1, 5)
    t2 = r.randint(1, 5)
    v1 = r.randint(30, 120)
    v2 = r.randint(30, 120)
    d1 = v1 * t1
    d2 = v2 * t2
    total_d = d1 + d2
    total_t = t1 + t2
    return {"d1": d1, "d2": d2, "t1": t1, "t2": t2, "total_d": total_d, "total_t": total_t, "speed": total_d / total_t}


def _gen_bulk_arith_nth(r: random.Random) -> Dict[str, Any]:
    a1 = r.randint(-10, 30)
    d = r.randint(1, 15)
    n = r.randint(4, 25)
    return {"a1": a1, "d": d, "n": n, "ans": a1 + (n - 1) * d}


def _gen_bulk_interest(r: random.Random) -> Dict[str, Any]:
    p = r.randint(100, 8000)
    rate = r.choice([2, 3, 4, 5, 6, 8, 10, 12])
    years = r.randint(1, 10)
    return {"p": p, "rate": rate, "years": years, "interest": p * rate * years / 100}


def _gen_bulk_ratio_split(r: random.Random) -> Dict[str, Any]:
    a = r.randint(1, 12)
    b = r.randint(1, 12)
    unit = r.randint(3, 50)
    parts = a + b
    total = unit * parts
    return {"a": a, "b": b, "parts": parts, "total": total, "unit": float(unit), "x": unit * a, "y": unit * b}


def _gen_bulk_unit_price(r: random.Random) -> Dict[str, Any]:
    qty = r.randint(2, 40)
    unit = r.choice([0.5, 0.75, 1.2, 1.5, 2.25, 2.8, 3.5, 4.75, 6.4, 8.0])
    total = qty * unit
    return {"qty": qty, "total": total, "unit": unit}


def _gen_bulk_weighted_avg(r: random.Random) -> Dict[str, Any]:
    hw = r.choice([20, 25, 30, 35, 40, 50, 60])
    exam = 100 - hw
    hw_score = r.randint(50, 100)
    exam_score = r.randint(40, 100)
    final = hw * hw_score / 100 + exam * exam_score / 100
    return {"hw": hw, "exam": exam, "hw_score": hw_score, "exam_score": exam_score, "final": final}


def _gen_bulk_all_heads(r: random.Random) -> Dict[str, Any]:
    n = r.randint(2, 12)
    denom = 2 ** n
    return {"n": n, "denom": denom, "prob": 1 / denom}


def _generate_bulk_math_templates(count: int = 1000) -> List[Dict[str, Any]]:
    theme_a = ["retail", "travel", "robotics", "ecology", "sports", "finance", "school", "manufacturing", "music", "gaming"]
    theme_b = ["practice", "workshop", "lab", "challenge", "exercise", "simulation", "drill", "review", "study set", "training"]
    themes = [f"{a} {b}" for a in theme_a for b in theme_b]  # 100 themes
    specs: List[Tuple[str, str, Any]] = [
        (
            "In a __THEME__ problem, solve for x: {a}x + {b} = {c}",
            "For this __THEME__ problem: subtract {b} from both sides to get {a}x = {rhs}. Then divide by {a}: x = {rhs}/{a} = {x:.2f}.",
            _gen_bulk_linear,
        ),
        (
            "In a __THEME__ scenario, a store offers a {p}% discount on an item priced at ${price}. What is the final price?",
            "Discount = ${price} x {p}/100 = ${disc:.2f}. Final price = ${price} - ${disc:.2f} = ${final:.2f}.",
            _gen_bulk_discount,
        ),
        (
            "In a __THEME__ worksheet, a rectangle has length {l} cm and width {w} cm. What are the area and perimeter?",
            "Area = {l} x {w} = {area}. Perimeter = 2 x ({l} + {w}) = {perim}.",
            _gen_bulk_rect,
        ),
        (
            "In a __THEME__ trip, a vehicle travels {d1} km in {t1} hours and then {d2} km in {t2} hours. What is the average speed?",
            "Total distance = {total_d} km, total time = {total_t} hours. Average speed = {total_d}/{total_t} = {speed:.1f} km/h.",
            _gen_bulk_avg_speed,
        ),
        (
            "In a __THEME__ sequence task, what is the {n}th term of an arithmetic sequence with first term {a1} and common difference {d}?",
            "Use a_n = a1 + (n-1)d. So a_{n} = {a1} + ({n}-1) x {d} = {ans}.",
            _gen_bulk_arith_nth,
        ),
        (
            "In a __THEME__ finance task, find the simple interest on ${p} at {rate}% per year for {years} years.",
            "Simple interest = P x r x t = {p} x {rate}/100 x {years} = ${interest:.2f}.",
            _gen_bulk_interest,
        ),
        (
            "In a __THEME__ ratio problem, divide ${total} in the ratio {a}:{b}.",
            "Total parts = {parts}. One part = {total}/{parts} = {unit:.2f}. Shares are ${x:.2f} and ${y:.2f}.",
            _gen_bulk_ratio_split,
        ),
        (
            "In a __THEME__ pricing task, if {qty} items cost ${total:.2f}, what is the unit price?",
            "Unit price = ${total:.2f}/{qty} = ${unit:.2f} per item.",
            _gen_bulk_unit_price,
        ),
        (
            "In a __THEME__ grade model, homework is {hw}% and exam is {exam}%. If homework is {hw_score}% and exam is {exam_score}%, what is the weighted average?",
            "Weighted grade = ({hw}/100) x {hw_score} + ({exam}/100) x {exam_score} = {final:.2f}%.",
            _gen_bulk_weighted_avg,
        ),
        (
            "In a __THEME__ probability drill, if you flip a fair coin {n} times, what is the probability of getting all heads?",
            "Each flip has probability 1/2 of heads. P(all heads) = (1/2)^{n} = 1/{denom} = {prob:.6f}.",
            _gen_bulk_all_heads,
        ),
    ]
    out: List[Dict[str, Any]] = []
    for theme in themes:
        for q_t, a_t, gen_fn in specs:
            out.append({
                "q": q_t.replace("__THEME__", theme),
                "a": a_t.replace("__THEME__", theme),
                "gen": gen_fn,
            })
            if len(out) >= count:
                return out
    return out


def _generate_bulk_logic_puzzles(count: int = 1000) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    syllables1 = ["ba", "zo", "tri", "lum", "fen", "kar", "nel", "vor", "sip", "mav"]
    syllables2 = ["lop", "rin", "tas", "vek", "mon", "dir", "zel", "nax", "pim", "qul"]
    for i in range(count):
        family = i % 10
        k = i // 10
        if family == 0:
            start = 2 + (k % 17)
            step = 2 + (k % 9)
            seq = [start + step * t for t in range(5)]
            q = f"What comes next in the sequence: {', '.join(map(str, seq))}, ...?"
            nxt = start + step * 5
            a = f"This is an arithmetic sequence with common difference {step}. The next term is {nxt}."
        elif family == 1:
            start = 1 + (k % 6)
            ratio = 2 + (k % 4)
            seq = [start * (ratio ** t) for t in range(4)]
            q = f"Find the next number: {', '.join(map(str, seq))}, ...?"
            nxt = start * (ratio ** 4)
            a = f"Each term is multiplied by {ratio}. The next number is {nxt}."
        elif family == 2:
            start = 20 + (k % 20)
            down = 2 + (k % 5)
            up = 1 + (k % 4)
            seq = [start]
            for _ in range(6):
                seq.append(seq[-1] - down if len(seq) % 2 == 1 else seq[-1] + up)
            q = f"Spot the pattern and give the next term: {', '.join(map(str, seq[:6]))}, ...?"
            nxt = seq[6]
            a = f"The pattern alternates -{down}, +{up}. Continuing gives {nxt}."
        elif family == 3:
            child = 6 + (k % 20)
            years = 2 + (k % 6)
            mult = 2 + (k % 3)
            diff = (mult - 1) * (child + years)
            q = (
                f"A parent is {diff} years older than a child. In {years} years, "
                f"the parent will be {mult} times the child's age. How old is the child now?"
            )
            a = (
                f"Let the child's age be c. Then the parent is c + {diff}. In {years} years: "
                f"c + {diff} + {years} = {mult}(c + {years}). Solving gives c = {child}."
            )
        elif family == 4:
            workers = 2 + (k % 9)
            hours = 2 + (k % 8)
            items = workers * hours
            new_workers = workers + 1 + (k % 7)
            q = (
                f"If {workers} identical workers make {items} units in {hours} hours at the same rate, "
                f"how long will {new_workers} workers take to make {new_workers * hours} units?"
            )
            a = (
                f"Each worker makes 1 unit per hour. {new_workers} workers make {new_workers} units per hour, "
                f"so to make {new_workers * hours} units they need {hours} hours."
            )
        elif family == 5:
            n = 4 + (k % 21)
            handshakes = n * (n - 1) // 2
            q = f"If {n} people each shake hands with every other person exactly once, how many handshakes occur?"
            a = f"The number of unique pairs is C({n},2) = {n} x {n-1} / 2 = {handshakes} handshakes."
        elif family == 6:
            h = (k % 12) + 1
            m = (k * 5) % 60
            hour_angle = (h % 12) * 30 + 0.5 * m
            minute_angle = 6 * m
            diff = abs(hour_angle - minute_angle)
            small = min(diff, 360 - diff)
            small_str = f"{small:.1f}".rstrip("0").rstrip(".")
            q = f"What is the smaller angle between the hands of a clock at {h}:{m:02d}?"
            a = f"Hour hand angle = {hour_angle:.1f} degrees, minute hand angle = {minute_angle:.1f} degrees. Smaller difference = {small_str} degrees."
        elif family == 7:
            reds = 1 + (k % 15)
            blues = 1 + ((k * 3) % 15)
            total = reds + blues
            prob = reds / total
            q = f"A bag has {reds} red marbles and {blues} blue marbles. What is the probability of drawing a red marble?"
            a = f"Probability = favorable / total = {reds}/{total} = {prob:.4f}."
        elif family == 8:
            x = syllables1[k % len(syllables1)] + syllables2[(k + 1) % len(syllables2)] + "s"
            y = syllables1[(k + 2) % len(syllables1)] + syllables2[(k + 3) % len(syllables2)] + "s"
            z = syllables1[(k + 4) % len(syllables1)] + syllables2[(k + 5) % len(syllables2)] + "s"
            if k % 2 == 0:
                q = f"All {x} are {y}. All {y} are {z}. Does it follow that all {x} are {z}?"
                a = "Yes. This is a valid transitive syllogism: if all x are y and all y are z, then all x are z."
            else:
                q = f"Some {x} are {y}. All {y} are {z}. Does it follow that all {x} are {z}?"
                a = "No. 'Some x are y' only tells us part of the x-group overlaps y, so it does not imply all x are z."
        else:
            n = 5 + (k % 8)
            k_choose = 2 + (k % min(4, n - 1))
            ways = math.comb(n, k_choose)
            q = f"How many different committees of {k_choose} can be chosen from {n} people?"
            a = f"This is a combinations problem: C({n},{k_choose}) = {ways}."
        out.append((q, a))
    return out


def _apply_requested_1000_seed_expansions() -> None:
    _append_exact_count(ANALOGY_DOMAINS, _generate_bulk_analogy_domains(1000), 1000, "ANALOGY_DOMAINS")
    _append_exact_count(DEBATE_TOPICS, _generate_bulk_debate_topics(1000), 1000, "DEBATE_TOPICS")
    _append_exact_count(STORY_GENRES, _generate_bulk_story_genres(1000), 1000, "STORY_GENRES")
    _append_exact_count(STORY_STARTERS, _generate_bulk_story_starters(1000), 1000, "STORY_STARTERS")
    _append_exact_count(MATH_TEMPLATES, _generate_bulk_math_templates(1000), 1000, "MATH_TEMPLATES")
    _append_exact_count(LOGIC_PUZZLES, _generate_bulk_logic_puzzles(1000), 1000, "LOGIC_PUZZLES")
    _append_exact_count(SOCRATIC_SEEDS, _generate_bulk_socratic_seeds(1000), 1000, "SOCRATIC_SEEDS")
    _append_exact_count(EMPATHY_SCENARIOS, _generate_bulk_empathy_scenarios(1000), 1000, "EMPATHY_SCENARIOS")
    _append_exact_count(CREATIVE_RESPONSE_PREFIXES, _generate_bulk_creative_prefixes(1000), 1000, "CREATIVE_RESPONSE_PREFIXES")
    _append_exact_count(REAL_CONVERSATIONS, _generate_bulk_real_conversations(1000), 1000, "REAL_CONVERSATIONS")


def _mod_tag(i: int) -> str:
    tags = [
        "advanced", "edge-case", "high-stakes", "real-world", "teaching", "systems",
        "practical", "strategic", "comparative", "reflective", "counterexample",
        "robustness", "creative", "analytical", "mentor", "debug", "research",
        "planning", "decision", "intuitive",
    ]
    return tags[i % len(tags)]


def _apply_second_1000_seed_expansions() -> None:
    base_analogies = _generate_bulk_analogy_domains(1000)
    ANALOGY_DOMAINS.extend([
        (f"{c} ({_mod_tag(i)} lens)", f"{a}, with an emphasis on {_mod_tag(i)} tradeoffs")
        for i, (c, a) in enumerate(base_analogies)
    ])

    base_debates = _generate_bulk_debate_topics(1000)
    DEBATE_TOPICS.extend([
        f"{topic} with a {_mod_tag(i)} policy framing"
        for i, topic in enumerate(base_debates)
    ])

    base_genres = _generate_bulk_story_genres(1000)
    STORY_GENRES.extend([
        f"{genre} ({_mod_tag(i)} variant)"
        for i, genre in enumerate(base_genres)
    ])

    base_starters = _generate_bulk_story_starters(1000)
    STORY_STARTERS.extend([
        f"{starter[:-1] if starter.endswith('.') else starter} Then the {_mod_tag(i)} witness arrived."
        for i, starter in enumerate(base_starters)
    ])

    base_math = _generate_bulk_math_templates(1000)
    MATH_TEMPLATES.extend([
        {
            "q": f"[{_mod_tag(i)} variant] {tpl['q']}",
            "a": f"[{_mod_tag(i)} reasoning] {tpl['a']}",
            "gen": tpl["gen"],
        }
        for i, tpl in enumerate(base_math)
    ])

    base_logic = _generate_bulk_logic_puzzles(1000)
    LOGIC_PUZZLES.extend([
        (
            f"[{_mod_tag(i)} puzzle] {q}",
            f"[{_mod_tag(i)} solution] {a}",
        )
        for i, (q, a) in enumerate(base_logic)
    ])

    base_socratic = _generate_bulk_socratic_seeds(1000)
    SOCRATIC_SEEDS.extend([
        (
            f"{q} ({_mod_tag(i)} angle)",
            f"{r} Also consider a {_mod_tag(i)} counterexample.",
            f"{ins} This variation emphasizes {_mod_tag(i)} reasoning habits.",
        )
        for i, (q, r, ins) in enumerate(base_socratic)
    ])

    base_empathy = _generate_bulk_empathy_scenarios(1000)
    EMPATHY_SCENARIOS.extend([
        (
            f"{s} I think the hardest part is the {_mod_tag(i)} pressure around it.",
            f"{resp} A {_mod_tag(i)} approach is to reduce the next step until it feels doable.",
        )
        for i, (s, resp) in enumerate(base_empathy)
    ])

    base_prefixes = _generate_bulk_creative_prefixes(1000)
    CREATIVE_RESPONSE_PREFIXES.extend([
        f"{p[:-1]} ({_mod_tag(i)}):" if p.endswith(":") else f"{p} ({_mod_tag(i)})"
        for i, p in enumerate(base_prefixes)
    ])


def _apply_third_10000_seed_expansions() -> None:
    def cyc_tag(i: int, cycle: int) -> str:
        return f"{_mod_tag(i + cycle * 97)}-set{cycle + 1}"

    base_analogies = _generate_bulk_analogy_domains(1000)
    analogies: List[Tuple[str, str]] = []
    for cycle in range(10):
        for i, (c, a) in enumerate(base_analogies):
            tag = cyc_tag(i, cycle)
            analogies.append(
                (
                    f"{c} [{tag}]",
                    f"{a}; focus on a {tag} explanation for beginners and advanced learners",
                )
            )
    _append_exact_count(ANALOGY_DOMAINS, analogies, 10000, "ANALOGY_DOMAINS_10K")

    base_debates = _generate_bulk_debate_topics(1000)
    debates: List[str] = []
    for cycle in range(10):
        for i, topic in enumerate(base_debates):
            tag = cyc_tag(i, cycle)
            debates.append(f"{topic} under a {tag} debate framing")
    _append_exact_count(DEBATE_TOPICS, debates, 10000, "DEBATE_TOPICS_10K")

    base_genres = _generate_bulk_story_genres(1000)
    genres: List[str] = []
    for cycle in range(10):
        for i, genre in enumerate(base_genres):
            tag = cyc_tag(i, cycle)
            genres.append(f"{genre} ({tag} genre variant)")
    _append_exact_count(STORY_GENRES, genres, 10000, "STORY_GENRES_10K")

    base_starters = _generate_bulk_story_starters(1000)
    starters: List[str] = []
    for cycle in range(10):
        for i, starter in enumerate(base_starters):
            tag = cyc_tag(i, cycle)
            base_text = starter[:-1] if starter.endswith(".") else starter
            starters.append(f"{base_text} Then the {tag} messenger arrived before dawn.")
    _append_exact_count(STORY_STARTERS, starters, 10000, "STORY_STARTERS_10K")

    base_math = _generate_bulk_math_templates(1000)
    math_templates: List[Dict[str, Any]] = []
    for cycle in range(10):
        for i, tpl in enumerate(base_math):
            tag = cyc_tag(i, cycle)
            math_templates.append(
                {
                    "q": f"[{tag} math variant] {tpl['q']}",
                    "a": f"[{tag} worked solution] {tpl['a']}",
                    "gen": tpl["gen"],
                }
            )
    _append_exact_count(MATH_TEMPLATES, math_templates, 10000, "MATH_TEMPLATES_10K")

    base_logic = _generate_bulk_logic_puzzles(1000)
    logic_puzzles: List[Tuple[str, str]] = []
    for cycle in range(10):
        for i, (q, a) in enumerate(base_logic):
            tag = cyc_tag(i, cycle)
            logic_puzzles.append((f"[{tag}] {q}", f"[{tag} reasoning] {a}"))
    _append_exact_count(LOGIC_PUZZLES, logic_puzzles, 10000, "LOGIC_PUZZLES_10K")

    base_socratic = _generate_bulk_socratic_seeds(1000)
    socratic_seeds: List[Tuple[str, str, str]] = []
    for cycle in range(10):
        for i, (q, r, ins) in enumerate(base_socratic):
            tag = cyc_tag(i, cycle)
            socratic_seeds.append(
                (
                    f"{q} [{tag}]",
                    f"{r} Push on a {tag} counterexample before finalizing your view.",
                    f"{ins} This variant adds {tag} reflection and comparison.",
                )
            )
    _append_exact_count(SOCRATIC_SEEDS, socratic_seeds, 10000, "SOCRATIC_SEEDS_10K")

    base_empathy = _generate_bulk_empathy_scenarios(1000)
    empathy_scenarios: List[Tuple[str, str]] = []
    for cycle in range(10):
        for i, (s, resp) in enumerate(base_empathy):
            tag = cyc_tag(i, cycle)
            empathy_scenarios.append(
                (
                    f"{s} The most difficult part is the {tag} pressure around it.",
                    f"{resp} A useful {tag} step is to lower the next action until it feels safe to begin.",
                )
            )
    _append_exact_count(EMPATHY_SCENARIOS, empathy_scenarios, 10000, "EMPATHY_SCENARIOS_10K")

    base_prefixes = _generate_bulk_creative_prefixes(1000)
    prefixes: List[str] = []
    for cycle in range(10):
        for i, p in enumerate(base_prefixes):
            tag = cyc_tag(i, cycle)
            if p.endswith(":"):
                prefixes.append(f"{p[:-1]} [{tag}]:")
            else:
                prefixes.append(f"{p} [{tag}]")
    _append_exact_count(CREATIVE_RESPONSE_PREFIXES, prefixes, 10000, "CREATIVE_RESPONSE_PREFIXES_10K")


if not globals().get("_MASSIVE_SEED_EXPANSIONS_APPLIED"):
    _apply_requested_1000_seed_expansions()
    _apply_second_1000_seed_expansions()
    _apply_third_10000_seed_expansions()
    _MASSIVE_SEED_EXPANSIONS_APPLIED = True

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _hash_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def generate_analogy_pairs(rng: random.Random, count: int) -> List[Dict[str, str]]:
    """Generate 'explain X like Y' pairs."""
    pairs = []
    templates = list(ANALOGY_DOMAINS)
    for i in range(count):
        concept, analogy = templates[i % len(templates)]
        prefix = rng.choice(CREATIVE_RESPONSE_PREFIXES)
        user = rng.choice([
            f"Can you explain {concept} using a simple analogy?",
            f"What's a good analogy for {concept}?",
            f"Help me understand {concept} in simple terms.",
            f"Explain {concept} like I'm a beginner.",
            f"Break down {concept} with a real-world comparison.",
        ])
        assistant = f"{prefix} {concept.capitalize()} is like {analogy}. "
        # Add elaboration
        elaborations = [
            f"Just as {analogy}, {concept} works on the same principle but in the digital/abstract world.",
            f"This analogy captures the essence: the core mechanism behind {concept} mirrors how {analogy}.",
            f"The parallel is instructive — understanding why {analogy} helps illuminate why {concept} behaves the way it does.",
        ]
        assistant += rng.choice(elaborations)
        pairs.append({"user": user, "assistant": assistant, "category": "analogy"})
    return pairs


def generate_debate_pairs(rng: random.Random, count: int) -> List[Dict[str, str]]:
    """Generate balanced pro/con argument pairs."""
    pairs = []
    for i in range(count):
        topic = DEBATE_TOPICS[i % len(DEBATE_TOPICS)]
        side = rng.choice(["for", "against"])
        user = rng.choice([
            f"Give me a strong argument {side} {topic}.",
            f"What's the case {side} {topic}?",
            f"Present the {side} argument for {topic}.",
            f"Why might someone argue {side} {topic}?",
        ])
        if side == "for":
            reasons = [
                "increases efficiency and well-being for most people",
                "addresses a real problem that existing solutions haven't solved",
                "aligns with principles of fairness and equal opportunity",
                "evidence from multiple countries/contexts supports positive outcomes",
            ]
        else:
            reasons = [
                "may create unintended consequences that outweigh the benefits",
                "infringes on individual autonomy or free-market principles",
                "existing alternatives are more practical and less disruptive",
                "implementation costs and logistical challenges are often underestimated",
            ]
        chosen_reasons = rng.sample(reasons, k=min(3, len(reasons)))
        prefix = rng.choice(CREATIVE_RESPONSE_PREFIXES)
        body = f"{prefix} The {side} case for {topic} rests on several pillars. "
        body += "First, " + chosen_reasons[0] + ". "
        if len(chosen_reasons) > 1:
            body += "Second, " + chosen_reasons[1] + ". "
        if len(chosen_reasons) > 2:
            body += "Finally, " + chosen_reasons[2] + "."
        body += f" That said, a thoughtful discussion of {topic} requires considering both sides carefully."
        pairs.append({"user": user, "assistant": body, "category": "debate"})
    return pairs


def generate_story_pairs(rng: random.Random, count: int) -> List[Dict[str, str]]:
    """Generate story-continuation prompts with varied genres."""
    pairs = []
    for i in range(count):
        genre = STORY_GENRES[i % len(STORY_GENRES)]
        starter = STORY_STARTERS[i % len(STORY_STARTERS)]
        user = rng.choice([
            f"Continue this {genre} story: {starter}",
            f"Write the next paragraph of this {genre} tale: {starter}",
            f"Here's a {genre} opening: '{starter}' — what happens next?",
        ])
        # Generate a continuation
        tones = {
            "science fiction": "technological wonder and cosmic scale",
            "fantasy": "magical atmosphere and mythical beings",
            "mystery": "suspense and hidden clues",
            "thriller": "heart-pounding tension and stakes",
            "romance": "emotional depth and connection",
            "historical fiction": "period-accurate detail and human drama",
            "horror": "creeping dread and atmospheric terror",
            "adventure": "bold action and discovery",
            "comedy": "wit, timing, and unexpected humor",
            "drama": "emotional complexity and character depth",
            "cyberpunk": "neon-lit streets and digital rebellion",
            "steampunk": "brass gears, steam, and Victorian ingenuity",
            "post-apocalyptic": "survival, desolation, and fragile hope",
            "fairy tale": "whimsy, moral lessons, and enchantment",
            "noir": "shadows, moral ambiguity, and sharp dialogue",
        }
        tone = tones.get(genre, "vivid storytelling")
        continuation = (
            f"[Continuing in the tradition of {genre}, with {tone}] "
            f"The moment hung in the air like a held breath. "
        )
        details = [
            "Something had changed — something fundamental — and there was no going back.",
            "The silence that followed was louder than any sound that came before.",
            "Every instinct screamed to turn away, but curiosity is a force stronger than fear.",
            "What came next would rewrite everything they thought they knew.",
            "It was the kind of moment that divides life into 'before' and 'after.'",
        ]
        continuation += rng.choice(details)
        pairs.append({"user": user, "assistant": continuation, "category": "storytelling"})
    return pairs


def generate_chain_of_thought_pairs(rng: random.Random, count: int) -> List[Dict[str, str]]:
    """Generate multi-step reasoning Q&A."""
    pairs = []
    # Math problems from templates
    for i in range(count // 2):
        template = MATH_TEMPLATES[i % len(MATH_TEMPLATES)]
        params = template["gen"](rng)
        q = template["q"].format(**params)
        a = template["a"].format(**params)
        user = rng.choice([
            q,
            f"Solve this step by step: {q}",
            f"Walk me through the solution: {q}",
        ])
        assistant = f"Let me work through this step by step. {a}"
        pairs.append({"user": user, "assistant": assistant, "category": "chain_of_thought"})

    # Logic puzzles
    for i in range(count - count // 2):
        puzzle_q, puzzle_a = LOGIC_PUZZLES[i % len(LOGIC_PUZZLES)]
        user = rng.choice([
            puzzle_q,
            f"Here's a puzzle: {puzzle_q}",
            f"Can you solve this logic question? {puzzle_q}",
        ])
        assistant = f"Let me reason through this carefully. {puzzle_a}"
        pairs.append({"user": user, "assistant": assistant, "category": "chain_of_thought"})
    return pairs


def generate_socratic_pairs(rng: random.Random, count: int) -> List[Dict[str, str]]:
    """Generate Socratic question-answer chains."""
    pairs = []
    for i in range(count):
        question, response, insight = SOCRATIC_SEEDS[i % len(SOCRATIC_SEEDS)]
        user = rng.choice([
            question,
            f"I've been thinking about this: {question}",
            f"Can you help me think through this question? {question}",
        ])
        prefix = rng.choice([
            "That's a profound question. ",
            "Great question — let's think about it together. ",
            "This is one of those questions that rewards deep thinking. ",
        ])
        assistant = f"{prefix}{response} {insight}"
        pairs.append({"user": user, "assistant": assistant, "category": "socratic"})
    return pairs


def generate_empathy_pairs(rng: random.Random, count: int) -> List[Dict[str, str]]:
    """Generate emotionally supportive responses."""
    pairs = []
    for i in range(count):
        scenario, response = EMPATHY_SCENARIOS[i % len(EMPATHY_SCENARIOS)]
        user = scenario
        assistant = response
        pairs.append({"user": user, "assistant": assistant, "category": "empathy"})
    return pairs


def generate_real_conversations_pairs(rng: random.Random, count: int) -> List[Dict[str, str]]:
    """Generate typical user support and real-world conversations."""
    pairs = []
    for i in range(count):
        scenario, response = REAL_CONVERSATIONS[i % len(REAL_CONVERSATIONS)]
        user = scenario
        assistant = response
        pairs.append({"user": user, "assistant": assistant, "category": "real_conversation"})
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate mega conversation dataset")
    parser.add_argument("--output", default="conversation_data.mega_creative.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--target", type=int, default=10000,
                        help="Target number of pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Distribute target across generators (roughly equal, with analogy getting slightly more)
    n_analogy = max(1, args.target * 22 // 100)
    n_debate = max(1, args.target * 18 // 100)
    n_story = max(1, args.target * 18 // 100)
    n_cot = max(1, args.target * 15 // 100)
    n_socratic = max(1, args.target * 10 // 100)
    n_real = max(1, args.target * 15 // 100)
    n_empathy = args.target - n_analogy - n_debate - n_story - n_cot - n_socratic - n_real

    print(f"Generating {args.target} pairs (analogy={n_analogy}, debate={n_debate}, "
          f"story={n_story}, cot={n_cot}, socratic={n_socratic}, real={n_real}, empathy={n_empathy})")

    all_pairs: List[Dict[str, str]] = []
    all_pairs.extend(generate_analogy_pairs(rng, n_analogy))
    all_pairs.extend(generate_debate_pairs(rng, n_debate))
    all_pairs.extend(generate_story_pairs(rng, n_story))
    all_pairs.extend(generate_chain_of_thought_pairs(rng, n_cot))
    all_pairs.extend(generate_socratic_pairs(rng, n_socratic))
    all_pairs.extend(generate_real_conversations_pairs(rng, n_real))
    all_pairs.extend(generate_empathy_pairs(rng, n_empathy))

    # Shuffle
    rng.shuffle(all_pairs)

    # Write JSONL
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_pairs)} pairs to {out_path}")
    print(f"Categories: {', '.join(sorted(set(p['category'] for p in all_pairs)))}")


if __name__ == "__main__":
    main()
