def data_analysis(data):
    lengths = [len(x) for x in data['AA_sequence']]
    letter_cnts = dict()
    for seq in data['AA_sequence']:
        for letter in seq:
            if letter not in letter_cnts:
                letter_cnts[letter] = 1
            else:
                letter_cnts[letter] += 1

    #plt.title('Raspodela duzina sekvenci amino kiselina')
    #plt.hist(lengths, bins=50)
    #plt.show()

    #cnt_different = len(letter_cnts.keys())

    # plt.title('Raspodela broja pojavljivanja svake kiseline')
    # plt.bar(letter_cnts.keys(), letter_cnts.values())
    #plt.show()

    longest_seq = max(lengths)
    print('Najduza sekvenca: ', longest_seq)
    categories = data['prot_Pfam'].unique()
    print('Broj razlicitih kategorija: ', len(categories))
