import re

# 1. Getting captions text data, cleaning and structuring it, creating vocabolury

# raw data
with open('Flicker8k_text/Flickr8k.token.txt') as file:
    text = file.readlines()


def img_captions_dict(filename):
    # creating dictionary of image_id and caption of all images  
    with open(filename) as file:
        text = file.readlines()
        
    captions_dict ={}
    
    for each_img_caption in text:
        img, caption = each_img_caption.split('\t')
        
        if img[:-2] not in captions_dict:
            captions_dict[img[:-2]] = [caption]
        else:
            captions_dict[img[:-2]].append(caption)
            
    return captions_dict


def cleaning_text(captions_dict):
    # cleaning and preprocessing of captions
    for img,caps in captions_dict.items():
        for i,img_caption in enumerate(caps):
            clean_caption = ' '.join(list(map(str.lower, filter(lambda x : len(x) > 1, filter(str.isalpha, re.sub(r'[^\w]+', " ",img_caption).replace('_', '').split())))))
            captions_dict[img][i] = clean_caption
            
    return captions_dict


def text_vocabulary(captions_dict):
    # build vocabulary of all unique words
    vocab = set()
    for key in captions_dict.keys():
        [vocab.update(d.split()) for d in captions_dict[key]]
    return vocab


def save_captions_in_one_file(captions_dict, filename):
    # saving all image id and captions in one text file
    lines = list()
    for key, caption_list in captions_dict.items():
        for cap in caption_list:
            lines.append(key + '\t' + cap )
    data = "\n".join(lines)

    with open(filename,"w") as file:
        file.write(data)


image_captions_dict = cleaning_text(img_captions_dict('Flicker8k_text/Flickr8k.token.txt'))
vocab = text_vocabulary(cleaning_text(img_captions_dict('Flicker8k_text/Flickr8k.token.txt')))
save_captions_in_one_file(image_captions_dict, 'captions.txt')