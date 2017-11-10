#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 This class contains most of the functionality needed for text cleaning!

 This code performs text normalization. Here I am performing Text Replacement
 (Contraction removal) and Smiley removal, we can replace
 contractions with their expanded versions. For example, doesn't can be
 replaced by does not and smilies like :), :D with "". Both are done with
 the help of Regex.
 
 Dealing with repeating characters and wrong spellings. Using wordnet
 along with it to ignore proper (real) english words
 This function is not going to work with sentences, so only words!!
 
 Dealing with repeating characters . Words involve
 repeating characters that cause grammatical errors. For instance
 consider a sentence, I like it lotttttt. Here, lotttttt refers to lot.
 so replacing lotttttt to lot.
 This function is not going to work with sentences, so only words!!

Created on Wed Aug 23 21:37:56 2017

@author: naanu
"""

import enchant
import re
from nltk.metrics import edit_distance
from nltk.corpus import wordnet


class UtilityClass(object):
    


    def __init__(self, dict_name='en', max_dist=2):
        
        self.contrList = {
        "ain't": 'am not',
        "aren't": 'are not',
        "can't": 'cannot',
        "can't've": 'cannot have',
        "'cause": 'because',
        "could've": 'could have',
        "couldn't": 'could not',
        "couldn't've": 'could not have',
        "didn't": 'did not',
        "doesn't": 'does not',
        "don't": 'do not',
        "hadn't": 'had not',
        "hadn't've": 'had not have',
        "hasn't": 'has not',
        "haven't": 'have not',
        "he'd": 'he would',
        "he'd've": 'he would have',
        "he'll": 'he will',
        "he'll've": 'he will have',
        "he's": 'he is',
        "how'd": 'how did',
        "how'd'y": 'how do you',
        "how'll": 'how will',
        "how's": 'how is',
        "I'd": 'I would',
        "I'd've": 'I would have',
        "I'll": 'I will',
        "I'll've": 'I will have',
        "I'm": 'I am',
        "I've": 'I have',
        "isn't": 'is not',
        "it'd": 'it had',
        "it'd've": 'it would have',
        "it'll": 'it will',
        "it'll've": 'it will have',
        "it's": 'it is',
        "let's": 'let us',
        "ma'am": 'madam',
        "mayn't": 'may not',
        "might've": 'might have',
        "mightn't": 'might not',
        "mightn't've": 'might not have',
        "must've": 'must have',
        "mustn't": 'must not',
        "mustn't've": 'must not have',
        "needn't": 'need not',
        "needn't've": 'need not have',
        "o'clock": 'of the clock',
        "oughtn't": 'ought not',
        "oughtn't've": 'ought not have',
        "shan't": 'shall not',
        "sha'n't": 'shall not',
        "shan't've": 'shall not have',
        "she'd": 'she would',
        "she'd've": 'she would have',
        "she'll": 'she will',
        "she'll've": 'she will have',
        "she's": 'she is',
        "should've": 'should have',
        "shouldn't": 'should not',
        "shouldn't've": 'should not have',
        "so've": 'so have',
        "so's": 'so is',
        "that'd": 'that would',
        "that'd've": 'that would have',
        "that's": 'that is',
        "there'd": 'there had',
        "there'd've": 'there would have',
        "there's": 'there is',
        "they'd": 'they would',
        "they'd've": 'they would have',
        "they'll": 'they will',
        "they'll've": 'they will have',
        "they're": 'they are',
        "they've": 'they have',
        "to've": 'to have',
        "wasn't": 'was not',
        "we'd": 'we had',
        "we'd've": 'we would have',
        "we'll": 'we will',
        "we'll've": 'we will have',
        "we're": 'we are',
        "we've": 'we have',
        "weren't": 'were not',
        "what'll": 'what will',
        "what'll've": 'what will have',
        "what're": 'what are',
        "what's": 'what is',
        "what've": 'what have',
        "when's": 'when is',
        "when've": 'when have',
        "where'd": 'where did',
        "where's": 'where is',
        "where've": 'where have',
        "who'll": 'who will',
        "who'll've": 'who will have',
        "who's": 'who is',
        "who've": 'who have',
        "why's": 'why is',
        "why've": 'why have',
        "will've": 'will have',
        "won't": 'will not',
        "won't've": 'will not have',
        "would've": 'would have',
        "wouldn't": 'would not',
        "wouldn't've": 'would not have',
        "y'all": 'you all',
        "y'alls": 'you alls',
        "y'all'd": 'you all would',
        "y'all'd've": 'you all would have',
        "y'all're": 'you all are',
        "y'all've": 'you all have',
        "you'd": 'you had',
        "you'd've": 'you would have',
        "you'll": 'you you will',
        "you'll've": 'you you will have',
        "you're": 'you are',
        "you've": 'you have',
        "i'd": 'I would',
        "i'd've": 'I would have',
        "i'll": 'I will',
        "i'll've": 'I will have',
        "i'm": 'I am',
        "i've": 'I have',
        }

        self.smileys = \
            """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^) :D 8-D 8D x-D xD X-D XD
                    =-D =D =-3 =3 B^D :D :\ :| ;( :( :/ :-( :'( :D :P :d :V :v""".split()
    
        self.countries = {
        'AF': 'Afghanistan',
        'AX': '\xc5land Islands',
        'AL': 'Albania',
        'DZ': 'Algeria',
        'AS': 'American Samoa',
        'AD': 'Andorra',
        'AO': 'Angola',
        'AI': 'Anguilla',
        'AQ': 'Antarctica',
        'AG': 'Antigua and Barbuda',
        'AR': 'Argentina',
        'AM': 'Armenia',
        'AW': 'Aruba',
        'AU': 'Australia',
        'AT': 'Austria',
        'AZ': 'Azerbaijan',
        'BS': 'Bahamas',
        'BH': 'Bahrain',
        'BD': 'Bangladesh',
        'BB': 'Barbados',
        'BY': 'Belarus',
        'BE': 'Belgium',
        'BZ': 'Belize',
        'BJ': 'Benin',
        'BM': 'Bermuda',
        'BT': 'Bhutan',
        'BO': 'Bolivia, Plurinational State of',
        'BQ': 'Bonaire, Sint Eustatius and Saba',
        'BA': 'Bosnia and Herzegovina',
        'BW': 'Botswana',
        'BV': 'Bouvet Island',
        'BR': 'Brazil',
        'IO': 'British Indian Ocean Territory',
        'BN': 'Brunei Darussalam',
        'BG': 'Bulgaria',
        'BF': 'Burkina Faso',
        'BI': 'Burundi',
        'KH': 'Cambodia',
        'CM': 'Cameroon',
        'CA': 'Canada',
        'CV': 'Cape Verde',
        'KY': 'Cayman Islands',
        'CF': 'Central African Republic',
        'TD': 'Chad',
        'CL': 'Chile',
        'CN': 'China',
        'CX': 'Christmas Island',
        'CC': 'Cocos (Keeling Islands)',
        'CO': 'Colombia',
        'KM': 'Comoros',
        'CG': 'Congo',
        'CD': 'Congo, The Democratic Republic of the',
        'CK': 'Cook Islands',
        'CR': 'Costa Rica',
        'CI': "D'ivoire",
        'HR': 'Croatia',
        'CU': 'Cuba',
        'CW': 'Cura',
        'CY': 'Cyprus',
        'CZ': 'Czech Republic',
        'DK': 'Denmark',
        'DJ': 'Djibouti',
        'DM': 'Dominica',
        'DO': 'Dominican Republic',
        'EC': 'Ecuador',
        'EG': 'Egypt',
        'SV': 'El Salvador',
        'GQ': 'Equatorial Guinea',
        'ER': 'Eritrea',
        'EE': 'Estonia',
        'ET': 'Ethiopia',
        'FK': 'Falkland Islands (Malvinas)',
        'FO': 'Faroe Islands',
        'FJ': 'Fiji',
        'FI': 'Finland',
        'FR': 'France',
        'GF': 'French Guiana',
        'PF': 'French Polynesia',
        'TF': 'French Southern Territories',
        'GA': 'Gabon',
        'GM': 'Gambia',
        'GE': 'Georgia',
        'DE': 'Germany',
        'GH': 'Ghana',
        'GI': 'Gibraltar',
        'GR': 'Greece',
        'GL': 'Greenland',
        'GD': 'Grenada',
        'GP': 'Guadeloupe',
        'GU': 'Guam',
        'GT': 'Guatemala',
        'GG': 'Guernsey',
        'GN': 'Guinea',
        'GW': 'Guinea-bissau',
        'GY': 'Guyana',
        'HT': 'Haiti',
        'HM': 'Heard Island and McDonald Islands',
        'VA': 'Holy See (Vatican City State)',
        'HN': 'Honduras',
        'HK': 'Hong Kong',
        'HU': 'Hungary',
        'IS': 'Iceland',
        'IN': 'India',
        'IND': 'India',
        'ID': 'Indonesia',
        'IR': 'Iran, Islamic Republic of',
        'IQ': 'Iraq',
        'IE': 'Ireland',
        'IM': 'Isle of Man',
        'IL': 'Israel',
        'IT': 'Italy',
        'JM': 'Jamaica',
        'JP': 'Japan',
        'JE': 'Jersey',
        'JO': 'Jordan',
        'KZ': 'Kazakhstan',
        'KE': 'Kenya',
        'KI': 'Kiribati',
        'KP': "Korea, Democratic People's Republic of",
        'KR': 'Korea, Republic of',
        'KW': 'Kuwait',
        'KG': 'Kyrgyzstan',
        'LA': 'Lao Democratic Republic',
        'LV': 'Latvia',
        'LB': 'Lebanon',
        'LS': 'Lesotho',
        'LR': 'Liberia',
        'LY': 'Libya',
        'LI': 'Liechtenstein',
        'LT': 'Lithuania',
        'LU': 'Luxembourg',
        'MO': 'Macao',
        'MK': 'Macedonia, The Former Yugoslav Republic of',
        'MG': 'Madagascar',
        'MW': 'Malawi',
        'MY': 'Malaysia',
        'MV': 'Maldives',
        'ML': 'Mali',
        'MT': 'Malta',
        'MH': 'Marshall Islands',
        'MQ': 'Martinique',
        'MR': 'Mauritania',
        'MU': 'Mauritius',
        'YT': 'Mayotte',
        'MX': 'Mexico',
        'FM': 'Micronesia, Federated States of',
        'MD': 'Moldova, Republic of',
        'MC': 'Monaco',
        'MN': 'Mongolia',
        'ME': 'Montenegro',
        'MS': 'Montserrat',
        'MA': 'Morocco',
        'MZ': 'Mozambique',
        'MM': 'Myanmar',
        'NA': 'Namibia',
        'NR': 'Nauru',
        'NP': 'Nepal',
        'NL': 'Netherlands',
        'NC': 'New Caledonia',
        'NZ': 'New Zealand',
        'NI': 'Nicaragua',
        'NE': 'Niger',
        'NG': 'Nigeria',
        'NU': 'Niue',
        'NF': 'Norfolk Island',
        'MP': 'Northern Mariana Islands',
        'NO': 'Norway',
        'OM': 'Oman',
        'PK': 'Pakistan',
        'PW': 'Palau',
        'PS': 'Palestinian Territory, Occupied',
        'PA': 'Panama',
        'PG': 'Papua New Guinea',
        'PY': 'Paraguay',
        'PE': 'Peru',
        'PH': 'Philippines',
        'PN': 'Pitcairn',
        'PL': 'Poland',
        'PT': 'Portugal',
        'PR': 'Puerto Rico',
        'QA': 'Qatar',
        'RE': 'R\xe9union',
        'RO': 'Romania',
        'RU': 'Russian Federation',
        'RW': 'Rwanda',
        'BL': 'Saint Barth\xe9lemy',
        'SH': 'Saint Helena, Ascension and Tristan Da Cunha',
        'KN': 'Saint Kitts and Nevis',
        'LC': 'Saint Lucia',
        'MF': 'Saint Martin (French Part)',
        'PM': 'Saint Pierre and Miquelon',
        'VC': 'Saint Vincent and the Grenadines',
        'WS': 'Samoa',
        'SM': 'San Marino',
        'ST': 'Sao Tome and Principe',
        'SA': 'Saudi Arabia',
        'SN': 'Senegal',
        'RS': 'Serbia',
        'SC': 'Seychelles',
        'SL': 'Sierra Leone',
        'SG': 'Singapore',
        'SX': 'Sint Maarten (Dutch Part)',
        'SK': 'Slovakia',
        'SI': 'Slovenia',
        'SB': 'Solomon Islands',
        'SO': 'Somalia',
        'ZA': 'South Africa',
        'GS': 'South Georgia and the South Sandwich Islands',
        'SS': 'South Sudan',
        'ES': 'Spain',
        'LK': 'Sri Lanka',
        'SD': 'Sudan',
        'SR': 'Suriname',
        'SJ': 'Svalbard and Jan Mayen',
        'SZ': 'Swaziland',
        'SE': 'Sweden',
        'CH': 'Switzerland',
        'SY': 'Syrian Arab Republic',
        'TW': 'Taiwan, Province of China',
        'TJ': 'Tajikistan',
        'TZ': 'Tanzania, United Republic of',
        'TH': 'Thailand',
        'TL': 'Timor-leste',
        'TG': 'Togo',
        'TK': 'Tokelau',
        'TO': 'Tonga',
        'TT': 'Trinidad and Tobago',
        'TN': 'Tunisia',
        'TR': 'Turkey',
        'TM': 'Turkmenistan',
        'TC': 'Turks and Caicos Islands',
        'TV': 'Tuvalu',
        'UG': 'Uganda',
        'UA': 'Ukraine',
        'AE': 'United Arab Emirates',
        'GB': 'United Kingdom',
        'UK': 'United Kingdom',
        'UK.': 'United Kingdom',
        'U.K': 'United Kingdom',
        'U.K.': 'United Kingdom',
        "US'": 'United States',
        'US': 'United States',
        'US.': 'United States',
        'U.S': 'United States',
        'U.S.A': 'United States America',
        'USA': 'United States America',
        'USA.': 'United States America',
        'UM': 'United States Minor Outlying Islands',
        'UY': 'Uruguay',
        'UZ': 'Uzbekistan',
        'VU': 'Vanuatu',
        'VE': 'Venezuela, Bolivarian Republic of',
        'VN': 'Viet Nam',
        'VG': 'Virgin Islands, British',
        'VI': 'Virgin Islands, U.S.',
        'WF': 'Wallis and Futuna',
        'EH': 'Western Sahara',
        'YE': 'Yemen',
        'ZM': 'Zambia',
        'ZW': 'Zimbabwe',
        }
        # spelling mistakes

        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

        # repeating words

        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

        self.contr_regex = re.compile('|'.join(self.contrList.keys()))
        self.smiley_regex = '|'.join(map(re.escape, self.smileys))
        self.country_regex = re.compile('|'.join(self.countries.keys()))


    # Special method called for example when you use print
    def __str__(self):
        return 'Trying to print an Object?? Check in Utility Class!!'

    

    def replaceSpelling(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)
        if suggestions and edit_distance(word, suggestions[0]) \
            <= self.max_dist:
            return suggestions[0]
        else:
            return word

    def replaceRepeat(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replaceRepeat(repl_word)
        else:
            return repl_word

    def expandContractions(self, text):

        def replace(match):
            return self.contrList[match.group(0)]

        return self.contr_regex.sub(replace, text)

    def removeSmileys(self, text):
        return re.sub(self.smiley_regex, '', text)

    def expandAbbreviations(self, text):

        def replace(match):
            return self.countries[match.group(0)]

        return self.country_regex.sub(replace, text)
    
    
    def testFunctionality(self):   

        print("Remove smiley: "+self.removeSmileys("hi how are you :D"))
        print("Replace spelling: "+self.replaceSpelling("lub"))
        print("Replace repeating alphabet: "+self.replaceRepeat("hellloo"))
        print("Expand contarction: "+self.expandContractions("hi i've"))
        print("Expland abbrevation: "+self.expandAbbreviations("hi US"))
        
        