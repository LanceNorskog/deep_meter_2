
import cmudict


class Decoder:
  
  def __init__(self, reverse_dict):
    self.reverse_dict = reverse_dict

  def decode(self, arpa_list):
    print(arpa_list)
    first = 0
    last = 1
    words = []
    while first < len(arpa_list):
      word = " ".join(arpa_list[first:last])
      print(word)
      if word in self.reverse_dict:
        print("{0} -> {1}".format(word, self.reverse_dict[word]))
        words.append(self.reverse_dict[word])
        first = last
        last = first + 1
      else:
        last = last + 1
    if first == len(arpa_list):
      return words
    else:
      return None

if __name__ == "__main__":
  (x, y, reverse_dict) = cmudict.load_dictionary()
  decoder = Decoder(reverse_dict)
  x = decoder.decode('DH AH S AH N L IH T AA N IH NG HH IY V IH NG OW V ER HH EH D'.split(' '))
  print(x)
#'AE N D AO L OW L IH M P AH S R IH NG Z W IH DH L AW D AH L AA R M Z'
#'AE N HH AH M B AH L CH IH R F AH L HH AE P IY L AH V IH NG B AE N D'
#'P ER EY D IH NG IH N AH K AA M M AH JH EH S T IH K EH R'
#'DH AH K AA M ER S AH V DH AH W ER L D W IH DH T AA N IY L IH M'
#'DH AH W EY T AH V Y IH R Z AO R W ER L D L IY K EH R Z DH AE T P R EH S'
#'K AH N JH EH K CH ER AH V DH AH P L UW M AH JH AE N D DH AH F AO R M'
#'AE N D HH AE N D IH N HH AE N D DH AH L AE F IH NG B EH L AE D R EH S'
#'AH N T IH L DH AE T AW ER DH AH W AO R F EH R L AE S T AH D DH EH R'
#'AE N D R AE M B L IH NG B R AE M B AH L B EH R IY Z P AH L P AE N D S W IY T'
