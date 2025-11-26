import os

# pick up template that has the most matching with the others

def align_sameLen(template,seq_list):
    # align 1 template vs a list of seq
    SCORES = {}
    for seq in seq_list:
        score, ideal_score = 0,0
        for t,s in zip(list(template),list(seq)):
            score += BLOSUM62[(t,s)]
            ideal_score += BLOSUM62[(t,t)] 
        score = score / ideal_score
        SCORES[seq] = score
    return SCORES

def align_blosum62(template,seq_list):
    max_score = 0
    for seq in seq_list:
        score, ideal_score = 0,0  
        s_mod=''
        for t,s in zip(list(template),list(seq)):
            score += BLOSUM62[(t,s)]
            ideal_score += BLOSUM62[(t,t)]  # score when all res in the template match
            if t==s: s_mod += s
            else: s_mod += '-'
        score = score / ideal_score
        max_score = max(max_score,score)
        #print(f'template:{template} seq: {seq} ideal_score: {ideal_score}/{ideal_score}=1 current_score: {score} max: {max_score} \n{template}\n{s_mod}')
    return max_score

def score_blosum62(template,seq):
    score, ideal_score = 0,0  
    for t,s in zip(list(template.upper()),list(seq.upper())):
        try:
            score += BLOSUM62[(t,s)]
            ideal_score += BLOSUM62[(t,t)]  # score when all res in the template match
        except:
            print('ERROR:',template,'---',seq)
    score = score / ideal_score
    return score


# MAFFT alignment
def mafft_pairwise(template,seq_list,mafft_path):
    aligned = []
    max_score = 0
    
    for seq in seq_list:
        with open('pair.in','w') as fo:
            fo.write(f'>template\n{template}\n')
            fo.write(f'>seq\n{seq}\n')
        
        #os.system(f'/home/chinh/miniforge3/bin/mafft --maxiterate 100 --localpair --quiet {cwd}/pair.in > {cwd}/pair.out')
        os.system(f'{mafft_path} --maxiterate 100 --globalpair --quiet pair.in > pair.out')
        
        tem = []
        with open('pair.out') as lines:
            for line in [line for line in lines if '>' not in line]:
                tem.append(line.strip())
        (template,seq) = tem
        score = score_blosum62(template,seq)
        max_score = max(max_score,score)
        aligned.append((seq,score))
        #print(f'template : {template}\nsequence:  {seq} => {score}')
    return max_score, aligned

def template_pickup(cdr_list):
    ALIGNED, SCORE = {}, {}
    for i,template in enumerate(cdr_list):
        seq_list = cdr_list[:i] + cdr_list[i+1:]  # remaining sequences
        #ALIGNED[template] = align_blosum62(template,seq_list)
        max_score, aligned = mafft_pairwise(template,seq_list)
        SCORE[template] = max_score
        ALIGNED[template] = aligned

        print(f'{i}\n{template}')
        for s in aligned:
            print(s[0],s[1])

    print('aligned:',ALIGNED,'\nScores:',SCORE)
    sel_template,_ = [(k,v) for k,v in SCORE.items() if v == max(SCORE.values())][0]
    return sel_template


def mafft_MSA(cdr_list,fname,mafft_path):
    ALIGNED = {}
    max_score = -100
    with open(f'{fname}_msa.in','w') as fo:
        for i,seq in enumerate(cdr_list):
            fo.write(f'>{i}#\n{seq}\n')
        
    try: 
        #os.system(f'/home/chinh/miniforge3/bin/mafft --maxiterate 1000 --globalpair --inputorder --clustalout --quiet {fname}_msa.in > {fname}_msa.out')
        os.system(f'{mafft_path} --maxiterate 1000 --globalpair --inputorder --clustalout --quiet {fname}_msa.in > {fname}_msa.out')  # imac
        #os.system(f'/Users/chinhsutran/miniforge3/bin/mafft --maxiterate 1000 --globalpair --inputorder --clustalout --quiet {fname}_msa.in > {fname}_msa.out')  # imac
        
        with open(f'{fname}_msa.out') as lines:
            for line in [line for line in lines if '#' in line]:
                aligned_seq = (line.split('#')[1]).strip()
                seq_id = line.split('#')[0]
                max_len = len(aligned_seq)
                try:
                    score = score_blosum62('A'*max_len,aligned_seq)*10 / max_len     # using AAAA...AA as reference to calculate scores
                    ALIGNED[int(seq_id)] = score
                    max_score = max(max_score,score)
                except:
                    print(f'aligned_seq: {aligned_seq},seq_id:{seq_id},max_len:{max_len},max_score:{max_score},score:{score}')
        
        #print(ALIGNED)
        sel_template = [cdr_list[int(k)] for k,v in ALIGNED.items() if v == max(ALIGNED.values())]
    except:
        sel_template = 'none'
        
    return max_score, sel_template, ALIGNED

# BLOSUM62 MATRIX
# using gap score from mafft
BLOSUM62 = {
    ('C','C'): 9, ('C','S'):-1, ('C','T'):-1, ('C','A'): 0, ('C','G'):-3, ('C','P'):-3, ('C','D'):-3, ('C','E'):-4, ('C','Q'):-3, ('C','N'):-3, ('C','H'):-3, ('C','R'):-3, ('C','K'):-3, ('C','M'):-1, ('C','I'):-1, ('C','L'):-1, ('C','V'):-1, ('C','W'):-2, ('C','Y'):-2, ('C','F'):-2, ('C','-'):-1.53,
    ('S','C'):-1, ('S','S'): 4, ('S','T'): 1, ('S','A'): 1, ('S','G'): 0, ('S','P'):-1, ('S','D'): 0, ('S','E'): 0, ('S','Q'): 0, ('S','N'): 1, ('S','H'):-1, ('S','R'):-1, ('S','K'): 0, ('S','M'):-1, ('S','I'):-2, ('S','L'):-2, ('S','V'):-2, ('S','W'):-3, ('S','Y'):-2, ('S','F'):-2, ('S','-'):-1.53,
    ('T','C'):-1, ('T','S'): 1, ('T','T'): 5, ('T','A'): 0, ('T','G'):-2, ('T','P'):-1, ('T','D'):-1, ('T','E'):-1, ('T','Q'):-1, ('T','N'): 0, ('T','H'):-2, ('T','R'):-1, ('T','K'):-1, ('T','M'):-1, ('T','I'):-1, ('T','L'):-1, ('T','V'):-1, ('T','W'):-2, ('T','Y'):-2, ('T','F'):-2, ('T','-'):-1.53,
    ('A','C'): 0, ('A','S'): 1, ('A','T'): 0, ('A','A'): 4, ('A','G'): 0, ('A','P'):-1, ('A','D'):-2, ('A','E'):-1, ('A','Q'):-1, ('A','N'):-2, ('A','H'):-2, ('A','R'):-1, ('A','K'):-1, ('A','M'):-1, ('A','I'):-1, ('A','L'):-1, ('A','V'):-1, ('A','W'):-3, ('A','Y'):-2, ('A','F'):-2, ('A','-'):-1.53,
    ('G','C'):-3, ('G','S'): 0, ('G','T'):-2, ('G','A'): 0, ('G','G'): 6, ('G','P'):-2, ('G','D'):-1, ('G','E'):-2, ('G','Q'):-2, ('G','N'): 0, ('G','H'):-2, ('G','R'):-2, ('G','K'):-2, ('G','M'):-3, ('G','I'):-4, ('G','L'):-4, ('G','V'):-3, ('G','W'):-2, ('G','Y'):-3, ('G','F'):-3, ('G','-'):-1.53,
    ('P','C'):-3, ('P','S'):-1, ('P','T'):-1, ('P','A'):-1, ('P','G'):-2, ('P','P'): 7, ('P','D'):-1, ('P','E'):-1, ('P','Q'):-1, ('P','N'):-2, ('P','H'):-2, ('P','R'):-2, ('P','K'):-1, ('P','M'):-2, ('P','I'):-3, ('P','L'):-3, ('P','V'):-2, ('P','W'):-4, ('P','Y'):-3, ('P','F'):-4, ('P','-'):-1.53,
    ('D','C'):-3, ('D','S'): 0, ('D','T'):-1, ('D','A'):-2, ('D','G'):-1, ('D','P'):-1, ('D','D'): 6, ('D','E'): 2, ('D','Q'): 0, ('D','N'): 1, ('D','H'):-1, ('D','R'):-2, ('D','K'):-1, ('D','M'):-3, ('D','I'):-3, ('D','L'):-4, ('D','V'):-3, ('D','W'):-4, ('D','Y'):-3, ('D','F'):-3, ('D','-'):-1.53,
    ('E','C'):-4, ('E','S'): 0, ('E','T'):-1, ('E','A'):-1, ('E','G'):-2, ('E','P'):-1, ('E','D'): 2, ('E','E'): 5, ('E','Q'): 2, ('E','N'): 0, ('E','H'): 0, ('E','R'): 0, ('E','K'): 1, ('E','M'):-2, ('E','I'):-3, ('E','L'):-3, ('E','V'):-2, ('E','W'):-3, ('E','Y'):-2, ('E','F'):-3, ('E','-'):-1.53,       
    ('Q','C'):-3, ('Q','S'): 0, ('Q','T'):-1, ('Q','A'):-1, ('Q','G'):-2, ('Q','P'):-1, ('Q','D'): 0, ('Q','E'): 2, ('Q','Q'): 5, ('Q','N'): 0, ('Q','H'): 0, ('Q','R'): 1, ('Q','K'): 1, ('Q','M'): 0, ('Q','I'):-3, ('Q','L'):-2, ('Q','V'):-2, ('Q','W'):-2, ('Q','Y'):-1, ('Q','F'):-3, ('Q','-'):-1.53,
    ('N','C'):-3, ('N','S'): 1, ('N','T'): 0, ('N','A'):-2, ('N','G'): 0, ('N','P'):-2, ('N','D'): 1, ('N','E'): 0, ('N','Q'): 0, ('N','N'): 6, ('N','H'): 1, ('N','R'): 0, ('N','K'): 0, ('N','M'):-2, ('N','I'):-3, ('N','L'):-3, ('N','V'):-3, ('N','W'):-4, ('N','Y'):-2, ('N','F'):-3, ('N','-'):-1.53,
    ('H','C'):-3, ('H','S'):-1, ('H','T'):-2, ('H','A'):-2, ('H','G'):-2, ('H','P'):-2, ('H','D'):-1, ('H','E'): 0, ('H','Q'): 0, ('H','N'): 1, ('H','H'): 8, ('H','R'): 0, ('H','K'):-1, ('H','M'):-2, ('H','I'):-3, ('H','L'):-3, ('H','V'):-3, ('H','W'):-2, ('H','Y'): 2, ('H','F'):-1, ('H','-'):-1.53,
    ('R','C'):-3, ('R','S'):-1, ('R','T'):-1, ('R','A'):-1, ('R','G'):-2, ('R','P'):-2, ('R','D'):-2, ('R','E'): 0, ('R','Q'): 1, ('R','N'): 0, ('R','H'): 0, ('R','R'): 5, ('R','K'): 2, ('R','M'):-1, ('R','I'):-3, ('R','L'):-2, ('R','V'):-3, ('R','W'):-3, ('R','Y'):-2, ('R','F'):-3, ('R','-'):-1.53,
    ('K','C'):-3, ('K','S'): 0, ('K','T'):-1, ('K','A'):-1, ('K','G'):-2, ('K','P'):-1, ('K','D'):-1, ('K','E'): 1, ('K','Q'): 1, ('K','N'): 0, ('K','H'):-1, ('K','R'): 2, ('K','K'): 5, ('K','M'):-1, ('K','I'): 3, ('K','L'):-2, ('K','V'):-2, ('K','W'):-3, ('K','Y'):-2, ('K','F'):-3, ('K','-'):-1.53,
    ('M','C'):-1, ('M','S'):-1, ('M','T'):-1, ('M','A'):-1, ('M','G'):-3, ('M','P'):-2, ('M','D'):-3, ('M','E'):-2, ('M','Q'): 0, ('M','N'):-2, ('M','H'):-2, ('M','R'):-1, ('M','K'):-1, ('M','M'): 5, ('M','I'): 1, ('M','L'): 2, ('M','V'): 1, ('M','W'):-1, ('M','Y'):-1, ('M','F'): 0, ('M','-'):-1.53,
    ('I','C'):-1, ('I','S'):-2, ('I','T'):-1, ('I','A'):-1, ('I','G'):-4, ('I','P'):-3, ('I','D'):-3, ('I','E'):-3, ('I','Q'):-3, ('I','N'):-3, ('I','H'):-3, ('I','R'):-3, ('I','K'):-3, ('I','M'): 1, ('I','I'): 4, ('I','L'): 2, ('I','V'): 3, ('I','W'):-3, ('I','Y'):-1, ('I','F'): 0, ('I','-'):-1.53,
    ('L','C'):-1, ('L','S'):-2, ('L','T'):-1, ('L','A'):-1, ('L','G'):-4, ('L','P'):-3, ('L','D'):-4, ('L','E'):-3, ('L','Q'):-2, ('L','N'):-3, ('L','H'):-3, ('L','R'):-2, ('L','K'):-2, ('L','M'): 2, ('L','I'): 2, ('L','L'): 4, ('L','V'): 1, ('L','W'):-2, ('L','Y'):-1, ('L','F'): 0, ('L','-'):-1.53,
    ('V','C'):-1, ('V','S'):-2, ('V','T'): 0, ('V','A'):-1, ('V','G'):-3, ('V','P'):-2, ('V','D'):-3, ('V','E'):-2, ('V','Q'):-2, ('V','N'):-3, ('V','H'):-3, ('V','R'):-3, ('V','K'):-2, ('V','M'): 1, ('V','I'): 3, ('V','L'): 1, ('V','V'): 4, ('V','W'):-3, ('V','Y'):-1, ('V','F'):-1, ('V','-'):-1.53,
    ('W','C'):-2, ('W','S'):-3, ('W','T'):-2, ('W','A'):-3, ('W','G'):-2, ('W','P'):-4, ('W','D'):-4, ('W','E'):-3, ('W','Q'):-2, ('W','N'):-4, ('W','H'):-2, ('W','R'):-3, ('W','K'):-3, ('W','M'):-1, ('W','I'):-3, ('W','L'):-2, ('W','V'):-3, ('W','W'):11, ('W','Y'): 2, ('W','F'): 1, ('W','-'):-1.53,
    ('Y','C'):-2, ('Y','S'):-2, ('Y','T'):-2, ('Y','A'):-2, ('Y','G'):-3, ('Y','P'):-3, ('Y','D'):-3, ('Y','E'):-2, ('Y','Q'):-1, ('Y','N'):-2, ('Y','H'): 2, ('Y','R'):-2, ('Y','K'):-2, ('Y','M'):-1, ('Y','I'):-1, ('Y','L'):-1, ('Y','V'):-1, ('Y','W'): 2, ('Y','Y'): 7, ('Y','F'): 3, ('Y','-'):-1.53,
    ('F','C'):-2, ('F','S'):-2, ('F','T'):-2, ('F','A'):-2, ('F','G'):-3, ('F','P'):-4, ('F','D'):-3, ('F','E'):-3, ('F','Q'):-3, ('F','N'):-3, ('F','H'):-1, ('F','R'):-3, ('F','K'):-3, ('F','M'): 0, ('F','I'): 0, ('F','L'): 0, ('F','V'):-1, ('F','W'): 1, ('F','Y'): 3, ('F','F'): 6, ('F','-'):-1.53,
    ('-','C'):-1.53,('-','S'):-1.53,('-','T'):-1.53,('-','A'):-1.53,('-','G'):-1.53,('-','P'):-1.53,('-','D'):-1.53,('-','E'):-1.53,('-','Q'):-1.53,('-','N'):-1.53,('-','H'):-1.53,('-','R'):-1.53,('-','K'):-1.53,('-','M'):-1.53,('-','I'):-1.53,('-','L'):-1.53,('-','V'):-1.53,('-','W'):-1.53,('-','Y'):-1.53,('-','F'):-1.53,('-','-'):-1.53,
    }