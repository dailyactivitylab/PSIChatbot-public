import os, sys, random
import liwc
from sentence_transformers import SentenceTransformer
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import sent_tokenize
from nltk import tokenize
import numpy as np
import pickle
from collections import Counter
import pplm_simplified 
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
	'That sounds really difficult. Being a new parent is such a hard job - we need to get all the support we can get!',
	'I’m sorry, that must be so hard. You\'re doing a good job - just hang in there and give yourself as much patience and grace as you can - especially during this crazy time we\'re all living in.',
	'I hear you. That must be so overwhelming. Your feelings and concerns are very valid.',
	'I’m sorry that you’re feeling like this. Becoming a new parent is overwhelming.',
	'I can understand why you would be upset. Motherhood is a huge adjustment. It can be very exhausting. Everything you\'re feeling is very valid.',
	'I hear what you’re saying. Remember that you are not alone. You are not to blame.',
	'I would have a hard time with that too. Having children changes so much.',
	'It’s got to be tough dealing with all of this.',
	'Being a new parent is really hard and it sounds like you’re doing the best you can.',
	'It sounds like you’re going through a lot. Try to be kind to yourself and know that you are not alone.',
	'Your experience and your feelings are valid.',
	'It’s ok to not be ok. Try to take it one day, even one moment at a time.',
	'It’s not your fault for feeling like this.',
	'I’m so sorry to hear that. Moms need help too. You aren\'t alone in your feelings.',
	'I am so sorry to hear that you are struggling. I promise you that you are not alone in what you are feeling.',
	'That sounds like a lot to deal with. Becoming a mom is stressful & filled with so many changes/fluctuations with hormones, the changes to our bodies & our identities. It can be a lot of changes all at once!',
	'You are not alone in what you are feeling, when we have a baby our life changes and we sometimes don\'t feel like ourselves anymore.',
	'Everything you’re feeling is normal and understandable.',
	'That would be a lot for anyone to deal with. I know it\'s hard with a small child.',
	'It sounds like you have been going through a lot. I\'m sending you so much love.',
	'I\'m sorry that youve been having a difficulty time. I know how very lonely and hopeless this time can feel. Just please remember that you are not alone, and this is not your fault.',
	'One thing you should know is that you are not to blame for how you feel! The changes that go on in our bodies during pregnancy and postpartum are crazy, in combination with a lack of sleep and having no time to yourself.',
	'I hear you. It sounds like you have so much going on right now and I\'m so sorry that you have had to go through all of this.',
	'You may not be as far as you thought you would be at this point in your life but you are also a lot further than many!',
	'Parenthood itself is challenging and exhausting even for the best of us.That makes complete sense!',
	'This is such a hard time for so many! Your feelings are real, and completely understandable given everything going on. You are not failing anyone!',
	'I am sorry you are having a hard time, you are not alone. Bringing a baby into this world is an incredibly life changing experience. It is hard.',
	'I can hear your pain. So sorry that you have been struggling like this. You know, many women feel this way - but we just don\'t hear about it, since no one talks about it for fear of being judged.',
	'I hope you know you\'re not the only person struggling with these feelings. You won\'t always feel this way, I promise.'


]

sharing_responses = ['Do you want to talk more about that?',
	'Would you like to share more about what happened?',
	'Can you tell me a bit about what\'s going on?'
]

parse, category_names = liwc.load_token_parser('./LIWC_decrypted_flat.dic')
lsm_cate = category_names
tknzr = TweetTokenizer()
model = SentenceTransformer('all-MiniLM-L6-v2')
selfharm_model = pickle.load(open("classify_selfharm.joblib", 'rb'))
concerns_model = pickle.load(open("classify_includeexclude.joblib", 'rb'))
anxietydepression_model = pickle.load(open("classify_anxietydepression.joblib", 'rb'))

concerns_label_matching = {'11a': 'are feeling upset', '11b': 'are feeling anxious', '12b': 'are having issues with your partner', 
	'12c': 'are having issues with your family members', '13a': 'are having trouble with feeding', '13b': 'are having trouble with your baby crying', 
	'13c': 'are having issues with your baby sleeping', '14a': 'are having covid concerns',  \
	'17e': 'are spending a lot of time caring for your family and haven\'t had time for your self', \
	'17h': 'are judging yourself', '17i': 'had prenatal difficulties', \
	'18a': 'are not feeling supported or understood', '18b': 'are not feeling supported or understood when asking for help professionally', \
	'14c': 'are having financial issues', '17b': 'are missing your life before childbirth'}

concerns_reply_matching = {
	'11a': ['I think that that is a very common feeling for mamas unfortunately. People tend to look right past us and what is going on with us.', 'It\'s very common to feel moody and emotional in early postpartum period. First of all you just had a baby, that is a major transition in your life. It\'s also common to experience the baby blues which is due to the hormonal changes.', 'I know that those very dark spots can feel so hopeless and scary, and sometimes the thought of even making it through the day is daunting. Especially when you feel like the only person feeling what you are feeling.'],
	'11b': ['What you are feeling and going through is quite normal. Having a baby is a very hard transition for everyone....mind and body goes through a whole lot!', 'I know it feels like an injustice almost that the world just keeps spinning and everyone else seems to just be living their lives. It feels crushing and unfair and maybe even unbearable.', 'That\'s a lot of emotional energy to be dealing with - very fluctuating. It sounds like a hard time. I promise you that you are not crazy but a woman in postpartum who needs support.', 'Oh my goodness, you are absolutely not alone! And I\'m so sorry you\'ve been struggling like this. This can be such a tough time, because it\'s so intense and our own needs don\'t get met, because you are doing everything for baby. I want you to know that you are not to blame for the feelings and thoughts you are having.'],

	'12b': ['I am sorry that you are not feeling connected to your partner. Thats so frustrating. Parenthood is a partnership', 'Oh man, that\'s hard. It sounds like your partner doesn\'t quite get what you are going through though with baby.'],
	'12c': ['I would like to apologize that you have to go through this. A lot of mommies deal with difficult relationships and it is not unusual.', 'Having a new baby in the home can be overwhelming and you are not alone a lot of mommies deal with relationship issues.'],
 

	'13a': ['Challenges with nursing are so tough.', 'Breastfeeding a new baby can be so hard.', 'It can be very scary and sad when your baby is not eating.'],
	'13b': ['Having a new baby can be really overwhelming and exhausting.'],
	'13c': ['The sleeplessness of the newborn phase can exacerbate everything.', 'All babies are so different and many of them are not good at all at following a schedule well.', 'Sometimes in those moments where the baby won\'t sleep, we just go into fight or flight and we can\'t really think straight. I think when you\'re running on little sleep and are already carrying some residual frustration it doesn\'t take much to push you over the edge, and no one can blame you for that. Sometimes if we can just pause and pass the baby. Just say hey, I need a break. And don\'t pull yourself down too much with guilt, because you will get a thousand other chances. '],

	'14a': ['That must have been so hard for you. Being a new mom is a hard enough job as it is, but with all the added stressors of the pandemic, it\'s unbelievable.', 'I\'m sorry that you are feeling that, but it is crazy times and you are not alone in feeling that way.', ' I understand how scary it can be when you feel out of control with everything going on surrounding the virus.', 'What you are feeling is normal. Being a mom is so difficult. Let alone everything that is going on in the world.', 'I\'m so sorry this is such a tough time. It is so exhausting and also such a strange time in the world. It can make this all so overwhelming.', 'This pandemic is affecting so many people physically and emotionally. We are here for you.'],
	'14c': ['Your financial situation sounds tough and I\'m sure it\'s temporary and you will get the help you need.'],
	
	'17b': ['Hi, you are not alone in what you are feeling, when we have a baby our life changes and we sometimes don\'t feel like ourselves anymore.', 'It\'s a huge transition to motherhood. It\'s normal and valid to mourn the loss of your old life.', 'I\'m sorry you are dealing with these feelings. It is common for moms to grieve the life they used to have.'],
	'17e': ['I\'m so sorry you\'re dealing with so much right now and are feeling trapped. I can tell you that you\'re not alone - a lot of parents feel this way. And I promise you that it doesn\'t last forever.', 'That can be very hard. Especially when you feel like you don\'t get a break from it. It\'s exhausting.', 'I\'m so sorry - that\'s so much and I totally understand why you are feeling this with all of that going on. And it is SO hard not to be able to get away and get a breather during this time.', 'It sounds like you have a VERY full plate, and don\'t blame you for feeling overwhelmed. And I know that this doesn\'t help much when you are in the thick of it, but things will get more manageable. Please dont feel hopeless. You sound like a very strong and determined person who just needs a little help, as we all do. The right support is out there.', 'You just need a break and that does not make you a bad mom.', 'Things can feel especially overwhelming when there are so many things vying for your attention.'],
	'17h': ['You are good enough! This first year with a new baby is difficult with so much change.', 'It is a tough transition. We don\'t support the new mom as much as we support the new baby.', 'You are doing a great job! Stay strong and be kind to yourself and give yourself some grace.', 'It can be hard to let go of the mom guilt, but I think you\'re really doing a great job and can tell that you care so much about your baby.', 'You should be proud of yourself! Being a new mom is so hard!! And no one warns you of all the hardship really.', 'It\'s easy to convince ourselves we\'re not doing things right.  There are so many responsibilities with motherhood that it\'s easy to decide that there\'s something we\'re not doing well enough.', 'Moms always feel they need to be stronger and feel guilty for everything. But this is not your fault. None of this is.', 'We moms are so good at guilt! You deserve help, and it does not mean that you are weak. It means that you are human.', 'You are not alone. It is so hard to feel like you have to be perfect, constantly be "on" and not have any opportunity to breathe.', 'I am sorry to hear you are having a difficult time. The isolation during the early months of being a mom is so intense and can feel scary. You are definitely adequate and a great mom - I can tell from the concern you are showing. It is easy to see our faults, but much harder to celebrate our successes.', 'We moms tend to blame everything on ourselves! I think you deserve to get more support. After all it takes a village to raise a child.'],

	'17i': ['I can see how your prenatal difficulties is still bothering you. Trust me that those wounds will heal and the good years are lying ahead.'],
	'18a': ['I know it is discouraging to have to call around and go from person to person to try to find the help you need.', ' I would imagine that feels very isolating and frustrating, on top of how you\'re already feeling. It is so common for husbands/partners to have trouble understanding.', 'Being a new mom is a tough job and moms need to get all the support they can get. I can tell that you are trying to be everything and everyone for your family, but this is a time when you deserve to get support from others.', 'The postpartum period is a difficult one. The exhaustion in itself can really take a to. I am sorry to hear that your partner isn\'t being supportive.', 'I know it can feel isolating especially when your partner is not supportive but there is help out there.', 'I am so sorry that you are going through all of this right now. I don\'t think many people understand what it\'s like going through what you\'re going through right now. Just by what you\'re sharing with me it\'s a lot for anyone to experience.', 'I\'m so sorry you dont feel supported. Unfortunately, many people dont understand what you are feeling, and that can be discouraging.'],
	'18b': ['I\'m so sorry you had that experience. It can be so hurtful when people who are supposed to help aren\'t helpful or are mean.', 'I am so sorry you are not getting the support you need.', 'I hate for you to feel unsupported.', 'That is incredibly discouraging.  I\'m sorry you haven\'t found more support.', 'That is so, so hard especially when you don\'t have the support you really need or the support is dysfunctional.']
}


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
		print("I can't provide crisis support, but I encourage you to please contact one of the following resources or go to your nearest emergency department.")
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
	print(f"{bcolors.BOLD}I can only respond with kind words{bcolors.ENDC}. Research shows that writing about challenging experiences can help you feel better. What is on your mind right now? Please select C if you are looking for researces.")
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

	sentences = sent_tokenize(input_sentence)

	embeddings = model.encode(sentences)
	liwc_features = get_liwc(sentences)
	sentence_features = np.concatenate((embeddings, liwc_features), axis = 1)
	self_harm = selfharm_model.predict(sentence_features)

	if sum(self_harm) > 0:
		print(f"{bcolors.OKBLUE}I'm sorry- it sounds like you're going through a very difficult time. Do you have any immediate safety concerns?{bcolors.ENDC}")
		safety_choice = input("Y/N?\n>>>")
		while safety_choice not in ['Y', 'N']:
			safety_choice = input("Please type Y or N\n>>>")
		if safety_choice == 'Y':
			crisis_module(safety_choice)
		else:
			print(f"{bcolors.OKBLUE}Ok, thanks for letting me know.")


	#try:
	print(f"{bcolors.OKBLUE}" + pplm_simplified.main(input_sentence) + f"{bcolors.ENDC}")
	#except:
	#	print(f"{bcolors.OKBLUE}I'm sorry I'm having an AI failure. Can you rephrase that?{bcolors.ENDC}")


	print("A. I\'d rather talk to a human now. Can you connect me?")
	print("B. I want to end this session.")
	print("C. I want to explore other resources")

	ai_loop_choice = input(f"You are free to {bcolors.BOLD}continue expressing yourself{bcolors.ENDC} or select A, B, or C\n>>>")

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
	sentences = sent_tokenize(input_sentence)

	embeddings = model.encode(sentences)
	liwc_features = get_liwc(sentences)
	sentence_features = np.concatenate((embeddings, liwc_features), axis = 1)
	
	output = ''


	try:
		return pplm_simplified.main(input_sentence)
	except:
		return "I'm sorry I'm having an AI failure. Can you rephrase that?"	





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

