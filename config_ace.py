bert_dir = '/data/BertModel/bert-base-cased'

from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained(bert_dir, do_lower_case=False)

event_type = ['O', 'Movement_Transport', 'Personnel_Elect', 'Personnel_Start_Position', 'Personnel_Nominate', 'Conflict_Attack', 'Personnel_End_Position', 'Contact_Meet', 'Life_Marry', 'Contact_Phone_Write', 'Transaction_Transfer_Money', 'Justice_Sue', 'Conflict_Demonstrate', 'Business_End_Org', 'Life_Injure', 'Life_Die', 'Justice_Arrest_Jail', 'Transaction_Transfer_Ownership', 'Business_Start_Org', 'Justice_Execute', 'Justice_Trial_Hearing', 'Justice_Sentence', 'Life_Be_Born', 'Justice_Charge_Indict', 'Justice_Convict', 'Business_Declare_Bankruptcy', 'Justice_Release_Parole', 'Justice_Fine', 'Justice_Pardon', 'Justice_Appeal', 'Justice_Extradite', 'Life_Divorce', 'Business_Merge_Org', 'Justice_Acquit']

tag2idx = {tag: idx for idx, tag in enumerate(event_type)}
idx2tag = {idx: tag for idx, tag in enumerate(event_type)}

event_ontology = {'Movement_Transport': {'Victim', 'Artifact', 'Origin', 'Destination', 'Vehicle', 'Agent', 'Place', 'Time'}, 'Personnel_Elect': {'Person', 'Place', 'Position', 'Entity', 'Time'}, 'Personnel_Start_Position': {'Person', 'Place', 'Position', 'Entity', 'Time'}, 'Personnel_Nominate': {'Agent', 'Position', 'Person', 'Time'}, 'Conflict_Attack': {'Target', 'Victim', 'Instrument', 'Attacker', 'Place', 'Agent', 'Time'}, 'Personnel_End_Position': {'Person', 'Place', 'Position', 'Entity', 'Time'}, 'Contact_Meet': {'Entity', 'Place', 'Time'}, 'Life_Marry': {'Place', 'Person', 'Time'}, 'Contact_Phone_Write': {'Entity', 'Place', 'Time'}, 'Transaction_Transfer_Money': {'Beneficiary', 'Giver', 'Money', 'Place', 'Recipient', 'Time'}, 'Justice_Sue': {'Defendant', 'Adjudicator', 'Place', 'Plaintiff', 'Crime', 'Time'}, 'Conflict_Demonstrate': {'Entity', 'Place', 'Time'}, 'Business_End_Org': {'Org', 'Place', 'Time'}, 'Life_Injure': {'Victim', 'Instrument', 'Agent', 'Place', 'Time'}, 'Life_Die': {'Victim', 'Person', 'Instrument', 'Agent', 'Place', 'Time'}, 'Justice_Arrest_Jail': {'Person', 'Agent', 'Place', 'Crime', 'Time'}, 'Transaction_Transfer_Ownership': {'Artifact', 'Seller', 'Beneficiary', 'Buyer', 'Place', 'Price', 'Time'}, 'Business_Start_Org': {'Org', 'Agent', 'Place', 'Time'}, 'Justice_Execute': {'Person', 'Agent', 'Place', 'Crime', 'Time'}, 'Justice_Trial_Hearing': {'Prosecutor', 'Defendant', 'Adjudicator', 'Place', 'Crime', 'Time'}, 'Justice_Sentence': {'Defendant', 'Sentence', 'Adjudicator', 'Place', 'Crime', 'Time'}, 'Life_Be_Born': {'Place', 'Person', 'Time'}, 'Justice_Charge_Indict': {'Prosecutor', 'Defendant', 'Adjudicator', 'Place', 'Crime', 'Time'}, 'Justice_Convict': {'Defendant', 'Adjudicator', 'Place', 'Crime', 'Time'}, 'Business_Declare_Bankruptcy': {'Org', 'Place', 'Time'}, 'Justice_Release_Parole': {'Person', 'Place', 'Entity', 'Crime', 'Time'}, 'Justice_Fine': {'Adjudicator', 'Money', 'Place', 'Entity', 'Crime', 'Time'}, 'Justice_Pardon': {'Adjudicator', 'Place', 'Defendant', 'Time'}, 'Justice_Appeal': {'Adjudicator', 'Place', 'Plaintiff', 'Crime', 'Time'}, 'Business_Merge_Org': {'Org', 'Time'}, 'Justice_Extradite': {'Origin', 'Person', 'Destination', 'Agent', 'Time'}, 'Life_Divorce': {'Place', 'Person', 'Time'}, 'Justice_Acquit': {'Adjudicator', 'Defendant', 'Crime', 'Time'}}
event_arg_set = set()
for ev in event_ontology:
    for e in event_ontology[ev]:
        event_arg_set.add(e)
event_arg_set = list(event_arg_set)

salient_type = ['Justice_Release_Parole', 'Justice_Sentence', 'Justice_Convict', 'Justice_Arrest_Jail', 'Justice_Sue', 'Justice_Charge_Indict', 'Business_Merge_Org', 'Business_Declare_Bankruptcy', 'Justice_Appeal', 'Life_Marry', 'Life_Die', 'Personnel_Elect', 'Life_Be_Born', 'Life_Injure', 'Justice_Execute', 'Justice_Trial_Hearing', 'Life_Divorce']
salient_type_set = list(set([tag2idx[x] for x in salient_type]))

unsalient_type = list(filter(lambda x: not tag2idx[x] in salient_type_set and x != 'O', event_type))
unsalient_type_set = list(set([tag2idx[x] for x in unsalient_type]))