import os, sys, random
import liwc
from sentence_transformers import SentenceTransformer
from nltk.tokenize.casual import TweetTokenizer
from nltk import tokenize
import numpy as np
import pickle
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


supportive_responses = ['That’s all very understandable.',
	'That sounds really difficult.',
	'I’m sorry, that must be so hard.',
	'That must be so overwhelming.',
	'I’m sorry that you’re feeling like this.',
	'I can understand why you would be upset.',
	'I hear what you’re saying.',
	'I would have a hard time with that too.',
	'It’s got to be tough dealing with all of this.',
	'Being a new parent is really hard and it sounds like you’re doing the best you can.',
	'It sounds like you’re going through a lot. Try to be kind to yourself and know that you are not alone.',
	'Your experience and your feelings are valid.',
	'It’s ok to not be ok. Try to take it one day, even one moment at a time.',
	'It’s not your fault for feeling like this.',
	'I’m so sorry to hear that.',
	'That sounds like a lot to deal with.',
	'Everything you’re feeling is so normal and understandable.',
	'That would be a lot for anyone to deal with.',
	'It sounds like you have been going through a lot.'
]

sharing_responses = ['Do you want to talk more about that?',
	'Tell me more about what happened?',
	'Can you tell me a bit about what\'s going on?',
	'What makes you feel this way?'
]

parse, category_names = liwc.load_token_parser('./LIWC_decrypted_flat.dic')
lsm_cate = category_names
tknzr = TweetTokenizer()
model = SentenceTransformer('all-MiniLM-L6-v2')
selfharm_model = pickle.load(open("classify_selfharm.joblib",'rb'))


def get_liwc(sentences):
	output = []
	for sentence in sentences:
		tokens = tknzr.tokenize(sentence)
		counts = Counter(category for token in tokens for category in parse(token))
		if sum(counts.values()) == 0:
			output.append([0] * len(lsm_cate))
		else:
			temp = []
			for item in lsm_cate:
				if item in counts.keys():
					temp.append(1)
				else:
					temp.append(0)
			output.append(temp)
	return np.array(output)


def start_module(start = False):
	if start == False:
		print("Hi there. I'm PSIbot. I'm a 24/7 chatbot, here to connect you to different resources. Select any option to learn more.")
	else:
		print("Ok! Click on any option to learn more.")
	print("A. I have immediate safety concerns.")
	print("B. Talk to a mental health professional.")
	print("C. I want to explore perinatal mental heatlh resources.")

	choice = input("A/B/C?\n>>>")
	while choice not in ['A', 'B', 'C']:
		choice = input("Please type A, B, or C\n>>>")

	if choice == 'A':
		crisis_module('Y')
	elif choice == 'B':
		professional_module()
	else:
		resources_module()
	return 


def end_module():
	#print("It was nice talking to you!")
	print("Reaching out for help is a brave step and I hope that you know how deserving you are of compassion and support.")
	print("Remember that you are not alone and you are not to blame. Help is available. You will get better.")
	sys.exit(0)


def crisis_module(safety_choice = None):
	if safety_choice == None:
		print("I'm sorry that you are going through such a difficult time, but I'm glad that you are reaching out for help. Do you have any immediate safety concerns?")
		safety_choice = input("Y/N?\n>>>")
		while safety_choice not in ['Y', 'N']:
			safety_choice = input("Please type Y or N\n>>>")

	if safety_choice == 'Y':
		print(f"{bcolors.BOLD}I can't provide crisis support{bcolors.ENDC}, but I encourage you to please contact one of the following resources or go to your nearest emergency department.")
		print("National Crisis Text Line: Text HOME to 741741 from anywhere in the USA, anytime, about any type of crisis.")
		print("National Suicide Prevention Hotline & Website: 1-800-273-8255 www.suicidepreventionlifeline.org")
		print("We want you to know you are not alone and you are not to blame. Help is available, and you will get better.")
		end_module()
	else:
		print("I'm glad to hear that. We have a 24-HR Hotline that you can call or text if you would like to talk to someone immediately.")
		print("Here are some additional resources:") 
		print("National Crisis Text Line: Text HOME to 741741 from anywhere in the USA, anytime, about any type of crisis.")
		print("National Suicide Prevention Hotline & Website: 1-800-273-8255 www.suicidepreventionlifeline.org")
		print("A. I'm done.")
		print("B. I want to learn about the hotline.")
		print("C. Connect me to the hotline.")
		print("D. I want to explore other resources.")
		no_crisis_choice = input("A/B/C/D?\n>>>")
		while no_crisis_choice not in ['A', 'B', 'C', 'D']:
			no_crisis_choice = input("Please type A, B, C, or D\n>>>")
		if no_crisis_choice == 'A':
			end_module()
		elif no_crisis_choice == 'B':
			helpline_submodule()
		elif no_crisis_choice == 'C':
			print("Call [insert hotline number] ")
			print("Text in English: 800-944-4773 \nText en Español: 971-203-7773")
			end_module()
		else:
			start_module(start = True)


	return

def professional_module():
	print("There are a few options for connecting with a professional. Select any option to learn more.")
	print("A. Call/text a (human) volunteer-operated helpline.")
	print("B. Search for a trained provider in your area using an online directory.")
	print("C. Share what is on your mind with me, PSIbot. I can listen now.")
	professional_choice = input("A/B/C?\n>>>")
	while professional_choice not in ['A', 'B', 'C']:
		professional_choice = input("Please type A, B, or C\n>>>")

	if professional_choice == 'A':
		helpline_submodule()
	elif professional_choice == 'B':
		directory_submodule()
	else:
		ai_submodule()
	return

def resources_module():
	print("We provide A. online support groups and B. education materials")
	resources_choice = input("A/B?\n>>>")
	while resources_choice not in ['A', 'B']:
		resources_choice = input("Please type A, or B\n>>>")

	if resources_choice == 'A':
		groups_submodule()
	else:
		materials_submodule()

	return

def helpline_submodule():
	print("I can connect you to a hotline or a helpline where a trained and caring volunteer will listen, answer questions, offer encouragement, and connect you with local resources as needed.")
	print("A. I want to explore other resources.")
	print("B. Hotline (immediate support)")
	print("C. I want to learn about the online directory instead")
	print("D. Helpline")
	helpline_choice = input("A/B/C/D?\n>>>")
	while helpline_choice not in ['A', 'B', 'C', 'D']:
		helpline_choice = input("Please type A, B, C or D\n>>>")

	if helpline_choice == 'A':
		start_module(start = True)
	elif helpline_choice == 'B':
		print("You can contact our 24-HR hotline if you would like to talk with someone immediately.")
		print("Call [insert hotline number] ")
		print("Text in English: 800-944-4773 \nText en Español: 971-203-7773")
		end_module()
	elif helpline_choice == 'C':
		directory_submodule()
	else:
		print("You can leave a confidential message at any time, and a volunteer will return your call or text during business hours.")
		print("Call 1-800-944-4773(4PPD)  #1 En Espanol or #2 English")
		print("Text in English: 800-944-4773 \nText en Español: 971-203-7773")
		end_module()
	return

def directory_submodule():
	print("By clickling the link below, you can search our extensive online directory to find providers and groups in your area. You can search by provider name, specialty, location, and any other specification.")
	print("[directory link]")
	print("A. I'm done.")
	print("B. Connect me to someone who can help me find providers in my area.")
	print("C. I want to explore other resources.")

	directory_choice = input("A/B/C?\n>>>")
	while directory_choice not in ['A', 'B', 'C']:
		directory_choice = input("Please type A, B, or C\n>>>")

	if directory_choice == 'A':
		end_module()
	elif directory_choice == 'B':
		helpline_submodule()
	else:
		start_module(start = True)
	return

def ai_submodule():
	print(f"{bcolors.BOLD}I can only respond with kind words.{bcolors.ENDC} Research shows that writing about challenging experiences can help you feel better. What is on your mind right now? Please select C if you are looking for researces.")
	print("A. I want to learn about the online directory instead.")
	print("B. I want to talk to a human.")
	print("C. I want to explore other resources")

	ai_choice = input(f"You are free to {bcolors.BOLD}type whatever is on your mind{bcolors.ENDC} or select A, B, or C\n>>>")

	if ai_choice == 'A':
		directory_submodule()
	elif ai_choice == 'B':
		helpline_submodule()
	elif ai_choice == 'C':
		start_module(start = True)
	else:
		ai_loop(ai_choice)

	return

	

def ai_loop(input_sentence):

	embeddings = model.encode([input_sentence])
	liwc_features = get_liwc([input_sentence])
	sentence_features = np.concatenate((embeddings, liwc_features), axis = 1)
	self_harm = selfharm_model.predict(sentence_features)[0]

	if self_harm == 1:
		print(f"{bcolors.OKBLUE}I'm sorry- it sounds like you're going through a very difficult time. Do you have any immediate safety concerns?{bcolors.ENDC}")

		safety_choice = input("Y/N?\n>>>")
		while safety_choice not in ['Y', 'N']:
			safety_choice = input("Please type Y or N\n>>>")
		if safety_choice == 'Y':
			crisis_module(safety_choice)
		else:
			print(f"{bcolors.OKBLUE}Ok, thanks for letting me know.{bcolors.ENDC}\n")

	print(f"{bcolors.OKBLUE}" + random.choice(supportive_responses) + f"{bcolors.ENDC}")

	print(f"{bcolors.OKBLUE}" + random.choice(sharing_responses) + f"{bcolors.ENDC}")

	print("A. I\'d rather talk to a human now. Can you connect me?")
	print("B. I want to end this session.")
	print("C. I want to explore other PSI resources")

	ai_loop_choice = input(f"You are free to {bcolors.BOLD}type whatever is on your mind{bcolors.ENDC} or select A, B, or C\n>>>")

	if ai_loop_choice == 'A':
		helpline_submodule()
	elif ai_loop_choice == 'B':
		end_module()
	elif ai_loop_choice == 'C':
		start_module(start = True)
	else:
		ai_loop(ai_loop_choice)

	return

def ai_loop_evaluate(input_sentence):
	return random.choice(supportive_responses) + ' ' + random.choice(sharing_responses)



def groups_submodule(groups_choice = None):
	if groups_choice == None:
		print("Online support groups allow you to take part in facillitated groups with others who can relate to what you are experiencing. Do you want to learn more about the group options?")
		print("A. Yes.")
		print("B. No.")
		print("C. I want to explore other resources.")

		groups_choice = input("A/B/C?\n>>>")
		while groups_choice not in ['A', 'B', 'C']:
			groups_choice = input("Please type A, B, or C\n>>>")

	if groups_choice == 'A':
		print("We have more than X free groups that you can register for. You can find all information by clicking this link: https://www.postpartum.net/get-help/psi-online-support-meetings/")
		print("Our groups offer support for a variety of concerns, such as:")
		print("A. Perinatal mood and anxiety disorder")
		print("B. Pregnancy, infant loss, and fertility challenges")
		print("C. Post-abortion")
		print("D. Support for family members")
		groups_choice_2 = input("Input A, B, C, or D for a list of available groups related to your concern.\n>>>")

		while groups_choice_2 not in ['A', 'B', 'C', 'D']:
			groups_choice_2 = input("Please type A, B, C, or D\n>>>")

		if groups_choice_2 in ['A', 'B', 'C', 'D']:
			print("(Example) Every Thursday @ 11 am ET/8 am PT")
			print("This group is intended for those who have been actively trying to conceive for six months or longer. It is led by PSI-trained facilitators (each with lived experience with primary and/or secondary fertility challenges).")
			print("Struggling with fertility issues can be a lonely experience and this group helps provide an avenue for healing and hope. This group helps women find support as they navigate the pain of fertility challenges.")


		print("What would yo like to do now?")
		print("A. I want to explore groups related to other concerns")
		print("B. I want to register for this group.")
		print("C. I'm done.")
		print("D. I want to explore other resources.")

		groups_choice_3 = input("A/B/C/D?\n>>>")
		while groups_choice_3 not in ['A', 'B', 'C', 'D']:
			groups_choice_3 = input("Please type A, B, C or D\n>>>")

		if groups_choice_3 == 'A':
			groups_submodule(groups_choice = 'A')
		elif groups_choice_3 == 'B':
			print("You can register by clicking this link: https://www.postpartum.net/get-help/psi-online-support-meetings/")
			end_module()
		elif groups_choice_3 == 'C':
			end_module()
		else:
			start_module(start = True)


	elif groups_choice == 'B':
		print("Would you like to learn more about perinatal mental health?")
		groups_choice_4 = input("Y/N?\n>>>")
		while groups_choice_4 not in ['Y', 'N']:
			groups_choice_4 = input("Please type Y or N\n>>>")

		if groups_choice_4 == 'Y':
			start_module(start = True)
		else:
			materials_submodule()
	else:
		start_module(start = True)

	return

def materials_submodule(start = None):
	if start == None:
		print("Did you know that 1 in 7 mothers and 1 in 10 fathers experience depression or anxiety during pregnancy or postpartum?")
		print("Here are some of the most common concerns of help seekers: ")
		print("- Feelings of guilt, shame, or hopelessness")
		print("- Anger, rage, irritability, or scary and unwanted thoughts")
		print("- Lack of interest in the baby or difficulty bonding with baby")
		print("- Loss of interest, joy, or pleasure in things you used to enjoy")
		print("- Disturbances in sleep and appetite")
		print("- Crying and sadness, constant worry, or racing thoughts")
		print("- Dizziness, hot flashes, nausea")
		print("- Possible thoughts of harming the baby or yourself")

	print("Although the term \"postpartum depression\" is often used, there are actually several overlapping illnesses. ")
	print("Select any option to learn more.")
	print("A. Depression")
	print("B. Anxiety")
	print("C. Obsessive-Compulsive Disorder")
	print("D. Postpartum stress disorder")
	print("E. Postpartum psychosis")
	print("F. Eating Disorders")
	print("G. Skip")

	materials_choice = input("A/B/C/D/E/F/G?\n>>>")
	while materials_choice not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
		materials_choice = input("Please type A, B, C, D, E, F, or G\n>>>")

	if materials_choice == 'A':  
		print("Pregnancy or Postpartum Depression might include feelings of anger, irritiability, guilt, lack of interest in the baby, changes ine ating and sleeping, trouble concerntrating, thoughts of hopelessness and sometimes thoughts of harming the baby or yourself")
	elif materials_choice == 'B':
		print("Pregnanncy or Postpartum Anxiety might include extreme worries and fears, including the health and safety of the baby. Some parents have panic attacks and might feel shortness of breath, chest pain, dizziness, feeling of losing control, numbness, and tingling.")
	elif materials_choice == 'C':
		print("Pregnancy or Postpartum Obsessive-Compulsive Disorder might include repetitive, upsetting, and unwanted thoguhts or mental images, and sometimes the need to do certain things over and over to reduce the anxiety caused by those thoughts. These parents find these thoughts very scary and unusual and are very unlikely to ever act on them.")
	elif materials_choice == 'D':
		print("Postpartum Stress Disorder is often caused by a traumatic or frightening childbirth. Symptoms might include flashbacks of the trauma with feelings of nxiety and the need to avoid things related to that event.")
	elif materials_choice == 'E':
		print("Postpartum Psychosis might include seeing or hearing voices or images toerh can't, feeling very energetic and unable to sleep, believing things that are not true and distrusting those around you. This rare illness can be dangerous so it is important to seek help immediatley.")
	elif materials_choice == 'F':
		print("Examples of Postpartum Eating Disorders include Anorexia Nervosa, Bulimia Nervosa, Binge Eating Disorder, and Orthorexia Nervosa. Symptoms might include excessive exercise, erratic eating patterns, hyper focus on changing body or weight-loss, preoccupation with food, isolation from loved ones, and negative body image. ")
	
	print("Things you can do to take care of yourself:")
	print("talk to a counselor or healthcare provider who has training in perniatal mood and anxiety problems")
	print("learn as much as you can about pregnancy and postpartum depression and anxiety")
	print("get support from family and friends")
	print("join a support group in your area or online")
	print("keep active by walking, stretching, or whatever form of movement helps you to feel better")
	print("get enought rest and time for yourself")
	print("fuel your body by eating regularly and including a variety of nutrients in your meals")

	print("\nConsult a health care provider if you have a history of or are currently experiencing symptoms of an eating disorder.")
	print("You can learn more by following this link: postpartum.net/resources")

	print("\nWhat would you like to do now?")
	print("A. I want to learn about another Postpartum Illness.")
	print("B. I want to learn about online support groups.")
	print("C. I'm done.")
	print("D. I want to explore other resources.")
	materials_choice_2 = input("A/B/C/D?\n>>>")
	while materials_choice_2 not in ['A', 'B', 'C', 'D']:
		materials_choice_2 = input("Please type A, B, C, or D\n>>>")

	if materials_choice_2 == 'A':
		materials_submodule(start = True)
	elif materials_choice_2 == 'B':
		groups_submodule()
	elif materials_choice_2 == 'C':
		end_module()
	else:
		start_module(start = True)

	return




if __name__ == "__main__":
	choice = start_module()

