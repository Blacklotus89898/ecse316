import struct
import socket

class DNSResponse:
    def __init__(self, response, query_name, server_ip, request_type):
        self.response = response
        self.query_name = query_name
        self.server_ip = server_ip
        self.request_type = request_type
        self.index = 0  # index to keep track of the current position in the response, like a pointer
        self.flags = 0

    def read_name(self, offset):
        labels = []
        while True:
            length = self.response[offset]
            if length & 0xC0 == 0xC0:  # Pointer to another location when compressed
                pointer = struct.unpack(">H", self.response[offset:offset + 2])[0]
                offset += 2
                return self.read_name(pointer & 0x3FFF)[0], offset
            if length == 0:  # End of name
                offset += 1
                break
            offset += 1
            labels.append(self.response[offset:offset + length].decode())
            offset += length
        return ".".join(labels), offset

    def parse_header(self):
        if len(self.response) < 12:
            raise ValueError("Invalid DNS response: too short")
        header = self.response[:12]
        (
            self.transaction_id,  # ID
            self.flags,  # |QR| Opcode |AA|TC|RD|RA| Z | RCODE |
            self.questions,  # QDCOUNT
            self.answer_rrs,  # ANCOUNT
            self.authority_rrs,  # NSCOUNT
            self.additional_rrs  # ARCOUNT
        ) = struct.unpack(">HHHHHH", header)
        self.index = 12

        # Check for errors in the response
        rcode = self.flags & 0x000F
        if rcode != 0:
            self.output_error(rcode)

    def parse_question(self):
        if self.questions != 1:
            raise ValueError(f"Invalid DNS response: expected 1 question, got {self.questions}")
        for _ in range(self.questions):
            _, self.index = self.read_name(self.index)
            self.index += 4  # Skip QTYPE and QCLASS

    def parse_record(self):
        name, self.index = self.read_name(self.index)
        if self.index + 10 > len(self.response):
            raise ValueError("Invalid DNS response: incomplete record section")
        answer_type, answer_class, ttl, data_length = struct.unpack(">HHIH", self.response[self.index:self.index + 10])
        self.index += 10
        if answer_class != 0x0001:
            raise ValueError(f"Invalid DNS response: expected CLASS 0x0001, got {answer_class:#04x}")
        if self.index + data_length > len(self.response):
            raise ValueError("Invalid DNS response: incomplete record data")
        auth = "auth" if (self.flags & 0x0400) else "nonauth"
        if answer_type == 1:  # Type A
            ip_address = socket.inet_ntoa(self.response[self.index:self.index + data_length])
            print(f"IP\t{ip_address}\t{ttl}\t{auth}")
        elif answer_type == 5:  # Type CNAME
            cname, _ = self.read_name(self.index)
            print(f"CNAME\t{cname}\t{ttl}\t{auth}")
        elif answer_type == 15:  # Type MX
            preference = struct.unpack(">H", self.response[self.index:self.index + 2])[0]
            mx, _ = self.read_name(self.index + 2)
            print(f"MX\t{mx}\t{preference}\t{ttl}\t{auth}")
        elif answer_type == 2:  # Type NS
            ns, _ = self.read_name(self.index)
            print(f"NS\t{ns}\t{ttl}\t{auth}")
        else:
            print("  Non-A record type")
        self.index += data_length

    def parse_response(self):
        self.parse_header()
        self.parse_question()
        if self.answer_rrs > 0:
            print(f"***Answer Section ({self.answer_rrs} records)***")
            for _ in range(self.answer_rrs):
                self.parse_record()
        else:
            print("NOTFOUND")
        if self.additional_rrs > 0:
            print(f"***Additional Section ({self.additional_rrs} records)***")
            for _ in range(self.additional_rrs):
                self.parse_record()

    def output_error(self, rcode):
        errors = {
            1: "ERROR    Format Error: the name server was unable to interpret the query.",
            2: "ERROR    Server Failure: the name server was unable to process this query due to failure with the name server.",
            3: "NOTFOUND",
            4: "ERROR    Not Implemented: the name server does not support that kind of query.",
            5: "ERROR    Refused: The server refused to answer for the query."
        }
        print(errors.get(rcode, "ERROR    Unknown error"))
        exit(0)

def parse_dns_response(response, query_name, server_ip, request_type):
    try:
        dns_response = DNSResponse(response, query_name, server_ip, request_type)
        dns_response.parse_response()
    except Exception as e:
        print(f"ERROR\t{str(e)}")