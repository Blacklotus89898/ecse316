'''
The following information was pulled from the "dnsprimer" document provided on the ECSE316 class on MyCourses.

Your application should be named DnsClient, and it should be invoked at the command line
using the following syntax:

python dnsClient.py [-t timeout] [-r max-retries] [-p port] [-mx|-ns] @server name

    • timeout (optional) gives how long to wait, in seconds, before retransmitting an
      unanswered query. Default value: 5.

    • max-retries(optional) is the maximum number of times to retransmit an unanswered
      query before giving up. Default value: 3.    python dnsClient.py -t 1 -r 1 @8.8.8.8 www.example.com

    • port (optional) is the UDP port number of the DNS server. Default value: 53.

    • -mx or -ns flags (optional) indicate whether to send a MX (mail server) or NS (name
      server) query. At most one of these can be given, and if neither is given then the client should
      send a type A (IP address) query.

    • server (required) is the IPv4 address of the DNS server, in a.b.c.d. format
    
    • name (required) is the domain name to query for.
'''

import socket
import argparse

from query import create_query_packet


# Helper function to parse the arguments set in the command line call
# (inspired by https://docs.python.org/3/library/argparse.html)
def parse_command_call():
    parser = argparse.ArgumentParser()  
    parser.add_argument("-t", default=5, dest="timeout", required=False, type=int)
    parser.add_argument("-r", default=3, dest="max_retries", required=False, type=int)
    parser.add_argument("-p", default=53, dest="port", required=False, type=int)
    parser.add_argument("-mx", action="store_true", required=False)
    parser.add_argument("-ns", action="store_true", required=False)
    parser.add_argument("server")
    parser.add_argument("name")

    args = parser.parse_args()

    query_type = "A"
    if args.mx:
        query_type = "MX"
    elif args.ns:
        query_type = "NS"

    server = args.server[1:]    # Ignore the leading @

    return args.timeout, args.max_retries, args.port, query_type, server, args.name


# (inspired by: https://docs.python.org/3/library/socket.html#functions)
# (inspired by: https://docs.python.org/3/library/socket.html)
# (inspired by: https://stackoverflow.com/questions/13999393/python-socket-sendto)
from response import parse_dns_response
import time
def send_query():
    timeout, max_retries, port, query_type, server, domain_name = parse_command_call()

    query_packet = create_query_packet(domain_name, query_type)

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.settimeout(timeout)
    retries = 0
    while (retries < max_retries):
        try:
            udp_socket.sendto(query_packet, (server, port))
            startTime = time.time_ns()
            response, serverIP = udp_socket.recvfrom(1024)
            if response:
                endTime = time.time_ns()
                udp_socket.close()
                print(f'DnsClient sending request for {domain_name}')
                print(f'Server: {serverIP}')
                print(f'Request type: {query_type}')
                print(f'Response received after {(endTime - startTime)/1000000000} seconds ({retries} retries)')
                parse_dns_response(response, domain_name, server, query_type)
                break

        except Exception as e:
            print(f"Error occurred: {e}")
            retries += 1

    if retries == max_retries:
        print(f"ERROR: Maximum retries reached ({retries})")
    udp_socket.close()


def main():
    '''
    # Test the query with the example provided in the dnsprimer doc
    domain_name = 'www.mcgill.ca'
    query_type = 'A'
    dns_query = create_query_packet(domain_name, query_type)
    print(f"DNS Query Packet: {dns_query.hex()}")
    '''

    # Test the parser for the command line arguments
    # timeout, max_retries, port, query_type, server, domain_name = parse_command_call()
    # print(f"Timeout: {timeout}")
    # print(f"Max retries: {max_retries}")
    # print(f"Port: {port}")
    # print(f"Query type: {query_type}")
    # print(f"Server: {server}")
    # print(f"Domain: {domain_name}")


    # Test the sending of the query
    send_query()







if __name__ == "__main__":
    main()