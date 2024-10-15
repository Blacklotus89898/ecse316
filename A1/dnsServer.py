import socket
import struct
import signal
import sys
import time

class SimpleDNSServer:
    def __init__(self, ip='127.0.0.1', port=8080, response_ip='192.168.1.1'):
        self.ip = ip
        self.port = port
        self.response_ip = response_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.running = True
        self.retry_count = {}
        print(f"DNS Server started on {self.ip}:{self.port}")

    def run(self):
        signal.signal(signal.SIGINT, self.shutdown)
        try:
            while self.running:
                data, addr = self.sock.recvfrom(512)
                if not data:
                    break
                print(f"Received request from {addr}")
                response = self.handle_request(data, addr)
                if response:
                    self.sock.sendto(response, addr)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.sock.close()

    def shutdown(self, signum, frame):
        print("\nServer is shutting down...")
        self.running = False
        self.sock.close()
        sys.exit(0)

    def handle_request(self, data, addr):
        transaction_id = data[:2]
        flags = struct.pack('>H', 0x8180)  # Standard query response, No error
        qdcount = struct.pack('>H', 1)  # Questions count
        ancount = struct.pack('>H', 1)  # Answer RRs count
        nscount = struct.pack('>H', 0)  # Authority RRs count
        arcount = struct.pack('>H', 1)  # Additional RRs count

        dns_header = transaction_id + flags + qdcount + ancount + nscount + arcount

        # Extract the question section
        question = data[12:]
        domain_name, qtype, qclass = self.parse_question(question)

        print(f"Parsed question: domain_name={domain_name}, qtype={qtype}, qclass={qclass}")

        # Simulate different error responses based on domain name
        if domain_name == "format.error":
            return self.create_error_response(transaction_id, 1)  # Format Error
        elif domain_name == "server.failure":
            return self.create_error_response(transaction_id, 2)  # Server Failure
        elif domain_name == "not.implemented":
            return self.create_error_response(transaction_id, 4)  # Not Implemented
        elif domain_name == "refused.query":
            return self.create_error_response(transaction_id, 5)  # Refused

        # Simulate delayed response
        if domain_name == "delayed.response":
            print("Simulating delayed response...")
            time.sleep(5)  # Delay for 5 seconds

        # Simulate response after retries
        if domain_name == "retry.response":
            if addr not in self.retry_count:
                self.retry_count[addr] = 0
            self.retry_count[addr] += 1
            print(f"Retry count for {addr}: {self.retry_count[addr]}")
            if self.retry_count[addr] < 3:
                print("Simulating no response for retry...")
                return None  # Do not respond until the third retry

        # Create the answer and additional sections
        answer = self.create_answer(domain_name, qtype, qclass)
        additional = self.create_additional_record()

        print(f"Generated answer: {answer.hex()}")
        print(f"Generated additional: {additional.hex()}")

        return dns_header + question + answer + additional

    def parse_question(self, question):
        domain_parts = []
        offset = 0
        while True:
            length = question[offset]
            if length == 0:
                break
            domain_parts.append(question[offset + 1:offset + 1 + length].decode())
            offset += length + 1
        domain_name = '.'.join(domain_parts)
        qtype, qclass = struct.unpack('>HH', question[offset + 1:offset + 5])
        return domain_name, qtype, qclass

    def create_answer(self, domain_name, qtype, qclass):
        answer_name = b'\xc0\x0c'  # Pointer to the domain name in the question section
        answer_type = struct.pack('>H', qtype)
        answer_class = struct.pack('>H', qclass)
        ttl = struct.pack('>I', 300)  # Time to live

        if qtype == 1:  # Type A
            rdlength = struct.pack('>H', 4)  # Length of the RDATA field
            rdata = socket.inet_aton(self.response_ip)  # The IP address to return
        elif qtype == 15:  # Type MX
            preference = struct.pack('>H', 10)  # Preference value
            exchange = self.compress_domain_name('mail.example.com')
            rdlength = struct.pack('>H', len(preference + exchange))
            rdata = preference + exchange
        elif qtype == 2:  # Type NS
            ns = self.compress_domain_name('ns.example.com')
            rdlength = struct.pack('>H', len(ns))
            rdata = ns
        else:
            rdlength = struct.pack('>H', 0)
            rdata = b''

        return answer_name + answer_type + answer_class + ttl + rdlength + rdata

    def create_additional_record(self):
        additional_name = b'\xc0\x0c'  # Pointer to the domain name in the question section
        additional_type = struct.pack('>H', 1)  # Type A
        additional_class = struct.pack('>H', 1)  # Class IN
        ttl = struct.pack('>I', 300)  # Time to live
        rdlength = struct.pack('>H', 4)  # Length of the RDATA field
        rdata = socket.inet_aton('192.168.1.2')  # The IP address to return for the additional record

        return additional_name + additional_type + additional_class + ttl + rdlength + rdata

    def compress_domain_name(self, domain_name):
        parts = domain_name.split('.')
        result = b''
        for part in parts:
            result += struct.pack('B', len(part)) + part.encode()
        result += b'\x00'
        return result

    def create_error_response(self, transaction_id, rcode):
        flags = struct.pack('>H', 0x8180 | rcode)  # Standard query response with error code
        qdcount = struct.pack('>H', 1)  # Questions count
        ancount = struct.pack('>H', 0)  # Answer RRs count
        nscount = struct.pack('>H', 0)  # Authority RRs count
        arcount = struct.pack('>H', 0)  # Additional RRs count

        dns_header = transaction_id + flags + qdcount + ancount + nscount + arcount
        return dns_header

if __name__ == '__main__':
    server = SimpleDNSServer(ip='127.0.0.1', port=8080, response_ip='192.168.1.1')
    server.run()