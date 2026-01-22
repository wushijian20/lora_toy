from datasets import load_dataset

# 1. children stories corpus
dataset = load_dataset("deven367/babylm-100M-children-stories")
print(dataset.shape)
print(dataset)
print(dataset['train'][0:2])
dataset_children_stories = dataset['train'].select(range(2000))
print(dataset_children_stories['text'][1])



# 2. bedtime stories
# dataset_1 = load_dataset("gofilipa/bedtime_stories")

# print(dataset_1.shape)
# print(dataset_1)
# print(dataset_1['train']['stories'][0:10])
# dataset_1 = dataset_1['train'].select(range(2000))

# 3. Shakespear corpus
# dataset_2 = load_dataset("sarnab/Shakespeare_Corpus")

# print(dataset_2.shape)
# print(dataset_2)
# print(dataset_2['train'][0:10])

# 4. King James Version Bible
# dataset_3 = load_dataset("SzuTao/KingJamesVersionBible")

# print(dataset_3.shape)
# print(dataset_3)
# print(dataset_3['train']['Text'][0:10])

# 5. Old English dataset

# dataset_4 = load_dataset("azizsi/old_english_dataset")
# print(dataset_4.shape)
# print(dataset_4)
# print(dataset_4['train']['Input'][0:10])
# print(dataset_4['train']['Output'][0:10])

# 6. Victorian authorship
# dataset_5 = load_dataset("contemmcm/victorian_authorship")

# print(dataset_5.shape)
# print(dataset_5)
# print(dataset_5['train']['text'][0:2])
# print(dataset_5['test']['text'][0:2])

# 7. Poetry chinese
# dataset_6 = load_dataset("erhwenkuo/poetry-chinese-zhtw")

# print(dataset_6.shape)
# print(dataset_6)
# print(dataset_6['train']['text'][0:2])
# print(dataset_6['test']['text'][0:2])

# 8. fairy_tales
# dataset_7 = load_dataset("aslicu/fairy_tales")

# print(dataset_7.shape)
# print(dataset_7)
# print(dataset_7['train']['chunk'][0:])


# 9. Mythological 
# dataset_8 = load_dataset("AJ69/Mythological")

# print(dataset_8.shape)
# print(dataset_8)
# print(dataset_8['train']['input'][0:2])
# print(dataset_8['train']['output'][0:2])

# 10. Mythological 
# dataset_9 = load_dataset("AJ69/Mythological")

# print(dataset_9.shape)
# print(dataset_9)
# print(dataset_9['train']['input'][0:2])
# print(dataset_9['train']['output'][0:2])

# 11. Casual Chat Pile 
# dataset_10 = load_dataset("Smilyai-labs/ChatPILE-Casual")

# print(dataset_10.shape)
# print(dataset_10)
# print(dataset_10['train']['messages'][0:2])

# 12. Casual Chat Pile 
# dataset_11 = load_dataset("phxdev/corporate-speak-dataset")

# print(dataset_11.shape)
# print(dataset_11)
# print(dataset_11['train']['output'][0:2])

# 13. 
# dataset_12 = load_dataset("sumukshashidhar-testing/research-paper-abstracts")

# 14. 
# dataset_13 = load_dataset("Samarth0710/neurips-2024-peer-reviews")

# 15. 
# dataset_14 = load_dataset("nvidia/Nemotron-Math-Proofs-v1")

# 16. 
# dataset_15 = load_dataset("emilpartow/reddit_finance_posts_sp500")

# 17. 
# dataset_16 = load_dataset("Thewillonline/reddit-sarcasm")

# 18. 
# dataset_17 = load_dataset("wenknow/reddit_dataset_44")



# 19. 
# dataset_18 = load_dataset("agentlans/reddit-logic")

# 20. 
# dataset_19 = load_dataset("cowWhySo/reddit_top_comments")

# 21. 
# dataset_20 = load_dataset("jonaskoenig/reddit-blogspot-twitter")

# 22. 
# dataset_21 = load_dataset("Osondu/reddit_autism_dataset")

# 23. 
# dataset_22 = load_dataset("Tlighteval/covid_dialogue")

# 24. 
# dataset_23 = load_dataset("sedthh/tv_dialogue")




# 25. 
# dataset_24 = load_dataset("Nexdata/American_English_Natural_Dialogue_Speech_Data")

# 26. 
# dataset_25 = load_dataset("erhwenkuo/medical_dialogue-chinese-zhtw")

# 27. 
# dataset_26 = load_dataset("jpeandrew/dialy_dialogue_with_recoginized_concept_raw")

# 28. 
# dataset_27 = load_dataset("rony/soccer-dialogues")

# 29. 
# dataset_28 = load_dataset("pixelsandpointers/empathetic_dialogues_for_lm") # Adapting/empathetic_dialogues_v2

# 30: Poems
dataset_30 = load_dataset("suayptalha/Poetry-Foundation-Poems") 
dataset_30 = dataset_30['train'].select(range(2000))
# print(dataset_30.shape)
# print(dataset_30)
# print(dataset_30['Poem'][0:2])





