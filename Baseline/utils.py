import torch
import smtplib


def send_email(sender_email, message, bot_email='pinakinathc.bot1@gmail.com', bot_pwd='pinakinathc1995'):
    smtpObj = smtplib.SMTP('smtp.gmail.com', 587) # Connect to google's smtp server
    smtpObj.ehlo()
    smtpObj.starttls()
    smtpObj.login(bot_email, bot_pwd)
    smtpObj.sendmail(bot_email, sender_email, f'Subject: Update from pinakinathc.bot-1\n\n{message}')

def collate_self(batch):
    batch_mod = {'img': [], 'sketch_boxes': [],
                 'label': [], 'sketch_path': []
                 }
    for i_batch in batch:
        batch_mod['img'].append(i_batch['img'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['label'].append(i_batch['label'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])

    batch_mod['img'] = torch.stack(batch_mod['img'], dim=0)
    batch_mod['label'] = torch.tensor(batch_mod['label'])

    return batch_mod