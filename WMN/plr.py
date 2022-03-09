import csv

ACTION = 0
TIME = 1
NODE = 2
LAYER = 3

start_observ = 10
sentPackets = 0
receivedPackets = 0

def sent_pack_by(row, node):
    return row[ACTION]=='s' and row[NODE] == node and row[LAYER] == 'AGT'


def rvc_pack_by(row, node):
    return row[ACTION]=='r' and row[NODE] == node and row[LAYER] == 'AGT'


with open('sanet.tr') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    line_count = 0
    for row in csv_reader:
        if float(row[TIME])>start_observ:
            if  sent_pack_by(row, '_0_'):
                sentPackets=sentPackets+1
            if rvc_pack_by(row, '_5_'):
                receivedPackets=receivedPackets+1


print(sentPackets, receivedPackets)

