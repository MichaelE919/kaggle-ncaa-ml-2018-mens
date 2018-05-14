"""
Uses Pandas, Request, and BeautifulSoup.

Parse HTML from kenpom.com and build a dataset containing the adjusted
offensive and defensive efficiencies and
the adjusted efficiency margin.
"""

import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

base_url = 'https://kenpom.com/index.php?y='


def swap(old_team, new_team):
    """Replace old_team with new_team for list n."""
    i = n.index(old_team)
    n.remove(old_team)
    n.insert(i, new_team)


print('Creating csv files for each year...')

csv_list = []
for year in range(2003, 2019):
    kp_page = requests.get(base_url + str(year))
    kp = kp_page.content

    soup = BeautifulSoup(kp, 'html.parser')

    td = soup.find_all('td')
    left = soup.find_all('td', {'class': 'td-left'})
    names = soup.find_all('a', {'href': re.compile('team\.php\?team=.+')})
    t = []
    s = []
    n = []

    for item in td:
        t.append(item.text)
    adjem_str = t[4::21]
    adjem = [float(i) for i in adjem_str]

    for number in left:
        s.append(float(number.text))

    for name in names:
        n.append(name.text)

    try:
        swap('Abilene Christian', 'Abilene Chr')
    except ValueError:
        pass
    swap('Alabama A&M;', 'Alabama A&M')
    swap('Alabama St.', 'Alabama St')
    swap('Albany', 'Albany NY')
    swap('Alcorn St.', 'Alcorn St')
    swap('American', 'American Univ')
    swap('Appalachian St.', 'Appalachian St')
    swap('Arizona St.', 'Arizona St')
    try:
        swap('Little Rock', 'Ark Little Rock')
    except ValueError:
        pass
    try:
        swap('Arkansas Little Rock', 'Ark Little Rock')
    except ValueError:
        pass
    swap('Arkansas Pine Bluff', 'Ark Pine Bluff')
    swap('Arkansas St.', 'Arkansas St')
    swap('Ball St.', 'Ball St')
    swap('Bethune Cookman', 'Bethune-Cookman')
    try:
        swap('Birmingham Southern', 'Birmingham So')
    except ValueError:
        pass
    swap('Boise St.', 'Boise St')
    swap('Boston University', 'Boston Univ')
    swap('Central Michigan', 'C Michigan')
    swap('Cal Poly', 'Cal Poly SLO')
    try:
        swap('Central Arkansas', 'Cent Arkansas')
    except ValueError:
        pass
    swap('Central Connecticut', 'Central Conn')
    swap('Charleston Southern', 'Charleston So')
    swap('Chicago St.', 'Chicago St')
    swap('The Citadel', 'Citadel')
    swap('Cleveland St.', 'Cleveland St')
    swap('Coastal Carolina', 'Coastal Car')
    swap('College of Charleston', 'Col Charleston')
    swap('Colorado St.', 'Colorado St')
    swap('Coppin St.', 'Coppin St')
    try:
        swap('Cal St. Bakersfield', 'CS Bakersfield')
    except ValueError:
        pass
    swap('Cal St. Fullerton', 'CS Fullerton')
    swap('Cal St. Northridge', 'CS Northridge')
    swap('Sacramento St.', 'CS Sacramento')
    swap('Delaware St.', 'Delaware St')
    swap('Eastern Illinois', 'E Illinois')
    swap('Eastern Kentucky', 'E Kentucky')
    swap('Eastern Michigan', 'E Michigan')
    swap('Eastern Washington', 'E Washington')
    try:
        swap('SIU Edwardsville', 'Edwardsville')
    except ValueError:
        pass
    swap('East Tennessee St.', 'ETSU')
    swap('Fairleigh Dickinson', 'F Dickinson')
    swap('Florida A&M;', 'Florida A&M')
    swap('Florida Atlantic', 'FL Atlantic')
    try:
        swap('Florida Gulf Coast', 'FL Gulf Coast')
    except ValueError:
        pass
    swap('FIU', 'Florida Intl')
    swap('Florida St.', 'Florida St')
    swap('Fresno St.', 'Fresno St')
    swap('George Washington', 'G Washington')
    swap('Georgia Southern', 'Ga Southern')
    swap('Georgia St.', 'Georgia St')
    swap('Grambling St.', 'Grambling')
    try:
        swap('Houston Baptist', 'Houston Bap')
    except ValueError:
        pass
    swap('Idaho St.', 'Idaho St')
    swap('Illinois Chicago', 'IL Chicago')
    swap('Illinois St.', 'Illinois St')
    swap('Indiana St.', 'Indiana St')
    swap('Iowa St.', 'Iowa St')
    try:
        swap('Fort Wayne', 'IPFW')
    except ValueError:
        pass
    swap('Jackson St.', 'Jackson St')
    swap('Jacksonville St.', 'Jacksonville St')
    swap('Kansas St.', 'Kansas St')
    try:
        swap('Kennesaw St.', 'Kennesaw')
    except ValueError:
        pass
    swap('Kent St.', 'Kent')
    swap('Long Beach St.', 'Long Beach St')
    swap('LIU Brooklyn', 'Long Island')
    swap('Loyola Marymount', 'Loy Marymount')
    swap('Loyola Chicago', 'Loyola-Chicago')
    try:
        swap('UMass Lowell', 'MA Lowell')
    except ValueError:
        pass
    swap('McNeese St.', 'McNeese St')
    swap('Maryland Eastern Shore', 'MD E Shore')
    swap('Michigan St.', 'Michigan St')
    swap('Mississippi St.', 'Mississippi St')
    swap('UMKC', 'Missouri KC')
    try:
        swap('Missouri St.', 'Missouri St')
    except ValueError:
        pass
    try:
        swap('Southwest Missouri St.', 'Missouri St')
    except ValueError:
        pass
    swap('Monmouth', 'Monmouth NJ')
    swap('Montana St.', 'Montana St')
    swap('Morehead St.', 'Morehead St')
    swap('Morgan St.', 'Morgan St')
    swap('Mississippi Valley St.', 'MS Valley St')
    swap('Mount St. Mary\'s', 'Mt St Mary\'s')
    try:
        swap('Middle Tennessee', 'MTSU')
    except ValueError:
        pass
    try:
        swap('Middle Tennessee St.', 'MTSU')
    except ValueError:
        pass
    swap('Murray St.', 'Murray St')
    try:
        swap('Northern Colorado', 'N Colorado')
    except ValueError:
        pass
    try:
        swap('North Dakota St.', 'N Dakota St')
    except ValueError:
        pass
    swap('Northern Illinois', 'N Illinois')
    try:
        swap('Northern Kentucky', 'N Kentucky')
    except ValueError:
        pass
    swap('North Carolina A&T;', 'NC A&T')
    try:
        swap('North Carolina Central', 'NC Central')
    except ValueError:
        pass
    swap('North Carolina St.', 'NC State')
    try:
        swap('Nebraska Omaha', 'NE Omaha')
    except ValueError:
        pass
    swap('New Mexico St.', 'New Mexico St')
    swap('Nicholls St.', 'Nicholls St')
    swap('Norfolk St.', 'Norfolk St')
    swap('Northwestern St.', 'Northwestern LA')
    swap('Ohio St.', 'Ohio St')
    swap('Oklahoma St.', 'Oklahoma St')
    swap('Oregon St.', 'Oregon St')
    swap('Penn St.', 'Penn St')
    swap('Portland St.', 'Portland St')
    swap('Prairie View A&M;', 'Prairie View')
    swap('South Carolina St.', 'S Carolina St')
    try:
        swap('South Dakota St.', 'S Dakota St')
    except ValueError:
        pass
    swap('Southern Illinois', 'S Illinois')
    swap('Sam Houston St.', 'Sam Houston St')
    swap('San Diego St.', 'San Diego St')
    swap('San Jose St.', 'San Jose St')
    swap('UC Santa Barbara', 'Santa Barbara')
    swap('Savannah St.', 'Savannah St')
    try:
        swap('USC Upstate', 'SC Upstate')
    except ValueError:
        pass
    swap('Southeastern Louisiana', 'SE Louisiana')
    swap('Southeast Missouri St.', 'SE Missouri St')
    swap('Stephen F. Austin', 'SF Austin')
    swap('Southern', 'Southern Univ')
    try:
        swap('Southwest Texas St.', 'Texas St')
    except ValueError:
        pass
    swap('St. Bonaventure', 'St Bonaventure')
    swap('St. Francis NY', 'St Francis NY')
    swap('St. Francis PA', 'St Francis PA')
    swap('St. John\'s', 'St John\'s')
    swap('Saint Joseph\'s', 'St Joseph\'s PA')
    swap('Saint Louis', 'St Louis')
    swap('Saint Mary\'s', 'St Mary\'s CA')
    swap('Saint Peter\'s', 'St Peter\'s')
    swap('Texas A&M;', 'Texas A&M')
    swap('Texas A&M; Corpus Chris', 'TAM C. Christi')
    swap('Tennessee St.', 'Tennessee St')
    try:
        swap('Texas St.', 'Texas St')
    except ValueError:
        pass
    swap('Tennessee Martin', 'TN Martin')
    try:
        swap('UT Rio Grande Valley', 'UTRGV')
    except ValueError:
        pass
    try:
        swap('Texas Pan American', 'UTRGV')
    except ValueError:
        pass
    swap('Texas Southern', 'TX Southern')
    try:
        swap('Troy St.', 'Troy')
    except ValueError:
        pass
    swap('Louisiana Lafayette', 'ULL')
    swap('Louisiana Monroe', 'ULM')
    swap('UTSA', 'UT San Antonio')
    swap('Utah St.', 'Utah St')
    try:
        swap('Utah Valley St.', 'Utah Valley')
    except ValueError:
        pass
    swap('VCU', 'VA Commonwealth')
    swap('Western Carolina', 'W Carolina')
    swap('Western Illinois', 'W Illinois')
    swap('Western Kentucky', 'WKU')
    swap('Western Michigan', 'W Michigan')
    swap('Washington St.', 'Washington St')
    swap('Weber St.', 'Weber St')
    try:
        swap('Winston Salem St.', 'W Salem St')
    except ValueError:
        pass
    swap('Green Bay', 'WI Green Bay')
    swap('Milwaukee', 'WI Milwaukee')
    swap('Wichita St.', 'Wichita St')
    swap('Wright St.', 'Wright St')
    swap('Youngstown St.', 'Youngstown St')

    adjo = s[0::8]
    adjd = s[1::8]

    df_kp = pd.DataFrame([n, adjo, adjd, adjem]).T
    columns = ['TeamName', 'AdjO', 'AdjD', 'AdjEM']
    df_kp.columns = columns
    df_kp.to_csv('KenPom_' + str(year) + '.csv', index=False)
    csv_list.append('KenPom_' + str(year) + '.csv')
    print('KenPom_' + str(year) + '.csv')

print('Combining yearly data into single csv file...')

# kp_2002 = pd.read_csv('KenPom_2002.csv')
kp_2003 = pd.read_csv('KenPom_2003.csv')
kp_2004 = pd.read_csv('KenPom_2004.csv')
kp_2005 = pd.read_csv('KenPom_2005.csv')
kp_2006 = pd.read_csv('KenPom_2006.csv')
kp_2007 = pd.read_csv('KenPom_2007.csv')
kp_2008 = pd.read_csv('KenPom_2008.csv')
kp_2009 = pd.read_csv('KenPom_2009.csv')
kp_2010 = pd.read_csv('KenPom_2010.csv')
kp_2011 = pd.read_csv('KenPom_2011.csv')
kp_2012 = pd.read_csv('KenPom_2012.csv')
kp_2013 = pd.read_csv('KenPom_2013.csv')
kp_2014 = pd.read_csv('KenPom_2014.csv')
kp_2015 = pd.read_csv('KenPom_2015.csv')
kp_2016 = pd.read_csv('KenPom_2016.csv')
kp_2017 = pd.read_csv('KenPom_2017.csv')
kp_2018 = pd.read_csv('KenPom_2018.csv')

# kp_2002.insert(0, column='Season', value=2002)
kp_2003.insert(0, column='Season', value=2003)
kp_2004.insert(0, column='Season', value=2004)
kp_2005.insert(0, column='Season', value=2005)
kp_2006.insert(0, column='Season', value=2006)
kp_2007.insert(0, column='Season', value=2007)
kp_2008.insert(0, column='Season', value=2008)
kp_2009.insert(0, column='Season', value=2009)
kp_2010.insert(0, column='Season', value=2010)
kp_2011.insert(0, column='Season', value=2011)
kp_2012.insert(0, column='Season', value=2012)
kp_2013.insert(0, column='Season', value=2013)
kp_2014.insert(0, column='Season', value=2014)
kp_2015.insert(0, column='Season', value=2015)
kp_2016.insert(0, column='Season', value=2016)
kp_2017.insert(0, column='Season', value=2017)
kp_2018.insert(0, column='Season', value=2018)

kp_total = pd.concat(
    [
        kp_2003, kp_2004, kp_2005, kp_2006, kp_2007, kp_2008, kp_2009, kp_2010,
        kp_2011, kp_2012, kp_2013, kp_2014, kp_2015, kp_2016, kp_2017, kp_2018
    ],
    ignore_index=True)

# Remove unnecessary cvs files
for file in csv_list:
    os.remove(file)

kp_total.to_csv('DataFiles/KenPom.csv', index=False)
