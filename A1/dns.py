'''
The following information was pulled from the "dnsprimer" document provided on the ECSE316 class on MyCourses.

DNS header:

       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      |                      ID                       | --> Random 16-bit identifier
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      |QR|   Opcode  |AA|TC|RD|RA|   Z    |   RCODE   |   = FLAGS
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      |                    QDCOUNT                    | --> 16-bit integer for the number of entries in the question section --> Always 1
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      |                    ANCOUNT                    | --> Unsigned 16-bit integer for the number of resource records in the answer section.
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      |                    NSCOUNT                    | --> Unsigned 16-bit integer for the number of name server resource records in the Authority section.
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      |                    ARCOUNT                    | --> Unsigned 16-bit integer specifying the number of resource records in the Additional records section.
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

  Flags:
    • QR is 1-bit
        0 - Query 
        1 - Response 
    • OPCODE is a 4-bit field that specifies the kind of query in this message. 
        --> Always 0 to represent a standard query.
    • AA is a bit only meaningful in responses
        0 - The name server is not an authority
        1 - The name server is an authority
    • TC is a bit only meaningful in responses
        Whether or not this message was truncated because it had a length greater than that permitted by the transmission channel.
    • RD is a bit set to indicate if server should pursue the query recursively. 
        --> Always 1 to indicate that you desire recursion.
    • RA is a bit only meaningful in responses
        Set or cleared by the server in a response message to indicate whether or not recursive queries are supported.
    • Z is a 3-bit field reserved for future use. Note: Your program should set this to 0.
    • RCODE is a 4-bit field, only meaningful in responses 
        0 - No error condition
        1 - Format error: the name server was unable to interpret the query
        2 - Server failure: the name server was unable to process this query due to a problem with the name server
        3 - Name error: meaningful only for responses from an authoritative name server, this code signifies that the domain name referenced in the query does not exist
        4 - Not implemented: the name server does not support the requested kind of query
        5 - Refused: the name server refuses to perform the requested operation for policy reasons
        --> Always 0 in request messages. 
'''
class DnsHeader:
  def __init__(self, ID, QR, OPCODE, AA, TC, RD, RA, Z, RCODE, QDCOUNT, ANCOUNT, NSCOUNT, ARCOUNT):
    self.ID = ID
    self.QR = QR
    self.OPCODE = OPCODE
    self.AA = AA
    self.TC = TC
    self.RD = RD
    self.RA = RA
    self.Z = Z
    self.RCODE = RCODE
    self.QDCOUNT = QDCOUNT
    self.ANCOUNT = ANCOUNT
    self.NSCOUNT = NSCOUNT
    self.ARCOUNT = ARCOUNT


  # Function to turn the header's fields into their byte representation
  def header_to_bytes(self):
    flags = (self.QR << 15) | (self.OPCODE << 11) | (self.AA << 10) | (self.TC << 9) | (self.RD << 8) | (self.RA << 7) | (self.Z << 4) | self.RCODE # Join all the bits to form the 2-byte FLAGS
    return (self.ID.to_bytes(2, 'big') +  
            flags.to_bytes(2, 'big') +   
            self.QDCOUNT.to_bytes(2, 'big') +  
            self.ANCOUNT.to_bytes(2, 'big') +  
            self.NSCOUNT.to_bytes(2, 'big') +  
            self.ARCOUNT.to_bytes(2, 'big'))



'''
The following information was pulled from the "dnsprimer" document provided on the ECSE316 class on MyCourses.

DNS question:

       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+ 
      |                                               | --> Domain name represented by a sequence of labels.
      |                    QNAME                      |     Each label begins with a length octet followed by that number of octets. 
      |                                               |     The domain name terminates with the zero-length octet.
      |                                               |
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+ 
      |                    QTYPE                      | --> 16-bit code specifying the type of query
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+ 
      |                    QCLASS                     | --> 16-bit code specifying the class of the query. --> Always use 0x0001 to represent an Internet address.
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    
    The three types relevant to QTYPE: 
      0x0001 for a type-A query (host address)
      0x0002 for a type-NS query (name server) 
      0x000f for a type-MX query (mail server)
'''
class DnsQuestion:
  def __init__(self, QNAME, QTYPE, QCLASS):
    self.QNAME = QNAME
    self.QTYPE = QTYPE
    self.QCLASS = QCLASS

  
  # Function to turn the question's fields into their byte representation
  def question_to_bytes(self):
    return (self.QNAME +  
            self.QTYPE.to_bytes(2, 'big') +  
            self.QCLASS.to_bytes(2, 'big'))