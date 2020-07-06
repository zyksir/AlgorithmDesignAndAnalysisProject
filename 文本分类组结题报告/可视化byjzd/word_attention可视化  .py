#!/usr/bin/env python
# coding: utf-8

# In[2]:


# coding: utf-8
def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(seq, attns):
    html = ""
    for ix, attn in zip(seq, attns):
        html += ' ' + highlight(
            ix,
            attn
        )
    return html + "<br>"

from IPython.display import HTML, display
import pickle
import numpy as np

def normalize(attn):
    ma = np.max(attn)
    mi = np.min(attn)
    for i in range(len(attn)):
        attn[i] = (attn[i] - mi)/(ma - mi)

with open('Data/kshfromjzd', 'rb') as f:
    data_z = pickle.load(f)
    length = len(data_z)
    seqs = data_z[int(length*0.9)+1:]
    
with open('Data/word_attn', 'rb') as f:
    data_z = pickle.load(f)
    lenx = len(data_z)
    leny = len(data_z[0])
    attns = np.array(data_z)
    attns = attns.reshape((lenx,leny))

with open('Data/out', 'rb') as f:
    out = pickle.load(f)

with open('Data/myproject_data', 'rb') as f:
    data_x, data_y = pickle.load(f)
    dev_y = data_y[int(length*0.9)+1:]
    
for k in range(len(seqs)):
    print(k)
    print(out[k])
    print(dev_y[k])
    for i in range(len(seqs[k])):
        normalize(attns[k*50+i])
        text = mk_html(seqs[k][i], attns[k*50 + i])
        display(HTML(text))


# In[ ]:





# In[ ]:




