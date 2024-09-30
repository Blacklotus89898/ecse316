import random

from dns import DnsHeader, DnsQuestion

def create_query(domain_name, query_type):
    # Create the header
    ID = random.getrandbits(16)
    QR = 0b0  # Signifies that the message is a query
    OPCODE = 0b0000 # To represent a standard query
    AA = 0b0    # N/A in queries
    TC = 0b0    # N/A in queries
    RD = 0b1    # To have the query pursue recursively
    RA = 0b0    # N/A in queries
    Z = 0b000   # N/A in queries
    RCODE = 0b000   # N/A in queries
    QDCOUNT = 0x0001    # Only one entry in the question
    ANCOUNT = 0x0000    # N/A in queries
    NSCOUNT = 0x0000    # N/A in queries
    ARCOUNT = 0x0000    # N/A in queries
    header = DnsHeader(ID, QR, OPCODE, AA, TC, RD, RA, Z, RCODE, QDCOUNT, ANCOUNT, NSCOUNT, ARCOUNT)

    # Create the question
    QNAME = build_name(domain_name)
    QTYPE = build_type(query_type)
    QCLASS = 0x0001 # Represents an Internet address
    question = DnsQuestion(QNAME, QTYPE, QCLASS)

    # Build the query
    return DnsHeader.header_to_bytes(header) + DnsQuestion.question_to_bytes(question)


# Helper function to convert a domain name into a byte string (inspired by https://implement-dns.wizardzines.com/book/part_1)
def build_name(domain_name):
    QNAME = b'' # An empty byte string
    labels = domain_name.split(".")
    for label in labels:
        label_length = len(label) 
        label_byte = bytes([label_length]) # The length of the label into a single byte
        label_ascii = label.encode('ascii') # Convert chars in label into their ASCII byte representation
        QNAME += label_byte + label_ascii
    QNAME += b"\x00"    # Signal the end of the domain name
    return QNAME


# Helper function to determine the query type
def build_type(query_type):
    if (query_type == "A"): return 0x0001
    elif (query_type == "NS"): return 0x0002
    elif (query_type == "MX"): return 0x000f