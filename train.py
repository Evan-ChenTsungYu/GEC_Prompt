import torch.nn as nn
from tqdm import tqdm
import torch
    

def train(model, TT_DL, Val_DL, epochs, result_save_path, tokenizer, show_epoch_result = True, DEVICE = 'cpu'):
    print(f'Train on {len(TT_DL)} sentence, with Validation set be {len(Val_DL)} sentence')
    start_epoch = 0 
    mini_val_loss = 1e10
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)
    
    for epoch in tqdm(range(start_epoch, epochs)):
        tot_loss = 0.0
        for index, content in enumerate(TT_DL):
            
            correct_text_label = content['prompt_label'].squeeze(1).to(DEVICE)
            correct_text_mask = content['prompt_mask'].squeeze(1).to(DEVICE)

            wrong_text_label = content['Data'].squeeze(1).to(DEVICE)
            wrong_text_mask = content['Data_Mask'].to(DEVICE)
            batch_size = wrong_text_label.size()[0]
            true_label = torch.ones((batch_size,1)).to(DEVICE)
            fake_label = torch.zeros((batch_size,1)).to(DEVICE)


            optimizer.zero_grad()
            output = model(wrong_sent = wrong_text_label,wrong_sent_mask = wrong_text_mask, correct_sent = correct_text_label)
            # output = model(input_ids = wrong_text_label, attention_mask = wrong_text_mask, labels = correct_text_label)
            # print(output)
            loss = output.loss

            loss.backward()
            optimizer.step()


            tot_loss += loss
        
        print(f'In each epoch check the output of decoder :')
        print(f'output : {tokenizer.decode(torch.argmax(output.logits, dim = 2)[0,:], skip_special_tokens = True)}')
        print(f'labels : {tokenizer.decode(correct_text_label[0,:], skip_special_tokens = True)}')
        # print(f' D_fake :{D_fake.squeeze(1)} fake_label :{fake_label} \n D_true :{D_true.squeeze(1)} true_label :{true_label}')
        # print(f'[{epoch}/ {epochs-1}], generator_loss:{tot_G_loss}, discriminator_loss:{tot_D_loss}')
    ## Val Dataset  ###############################################################################
        scheduler.step()
        scheduler.step()
        val_loss = 0
        print(f'Test Validation Loss, set val loss = {val_loss}')
        with torch.no_grad():
            for val_index, val_content in enumerate(Val_DL):
                # val_correct_text_label = val_content['classify_label'].squeeze(1).to(DEVICE)
                val_correct_text_label = val_content['prompt_label'].squeeze(1).to(DEVICE)
                val_correct_text_mask = val_content['prompt_mask'].squeeze(1).to(DEVICE)
                val_wrong_text_label = val_content['Data'].squeeze(1).to(DEVICE)
                val_wrong_input_mask = val_content['Data_Mask'].to(DEVICE)
        
                # D_fake, D_true = model(correct_text_label, correct_text_mask, wrong_text_label,wrong_text_mask, train_DGI = 'D')
                # output = model(wrong_sent = wrong_text_label,wrong_sent_mask = wrong_text_mask, correct_sent = correct_text_label)
                
                val_correcttext_generate = model(wrong_sent = val_wrong_text_label, wrong_sent_mask = val_wrong_input_mask, correct_sent = val_correct_text_label)
                # val_loss_dict = 
                # print(val_correcttext_generate.logits.shape)
                # print(f'val_correcttext_generate : {val_correcttext_generate}, val_correct_text_label:{val_correct_text_label}')
                val_loss += val_correcttext_generate.loss
                
                if val_index == 1:
                    # print(wrong_text_label[0,:20], torch.argmax(wrongtext,dim = 1)[0,:20])
                    decode_wrongtextlabel = tokenizer.decode(val_wrong_text_label[0,:])
                    decode_wrongtext = tokenizer.decode(torch.argmax(val_correcttext_generate.logits,dim = 1)[0,:])
                    # print(f'decode_wrongtextlabel:{decode_wrongtextlabel}\n decode_wrongtext:{decode_wrongtext[:10]}')
            
        if val_loss <= mini_val_loss:
            print(f'renew val_loss (Generator): {val_loss/len(Val_DL)}')
            mini_val_loss = val_loss
            torch.save(model.state_dict(), result_save_path)
        
        if show_epoch_result :
            print(f'[{epoch}/ {epochs-1}], loss:{tot_loss/len(TT_DL)} / val_loss:{val_loss/len(Val_DL)}') 
        # [print(f'lab: {label[i, :8]} \n out: {output[i, 1:9]}' ) for i in range(0, 5)]