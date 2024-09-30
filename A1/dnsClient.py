'''
Your application should be named DnsClient, and it should be invoked at the command line
using the following syntax:

python dnsClient.py [-t timeout] [-r max-retries] [-p port] [-mx|-ns] @server name

    • timeout (optional) gives how long to wait, in seconds, before retransmitting an
      unanswered query. Default value: 5.

    • max-retries(optional) is the maximum number of times to retransmit an unanswered
      query before giving up. Default value: 3.

    • port (optional) is the UDP port number of the DNS server. Default value: 53.

    • -mx or -ns flags (optional) indicate whether to send a MX (mail server) or NS (name
      server) query. At most one of these can be given, and if neither is given then the client should
      send a type A (IP address) query.

    • server (required) is the IPv4 address of the DNS server, in a.b.c.d. format
    
    • name (required) is the domain name to query for.
'''

from query import create_query

def main():
    # Test the query with the example provided in the dnsprimer doc
    domain_name = 'www.mcgill.ca'
    query_type = 'A'
    dns_query = create_query(domain_name, query_type)
    print(f"DNS Query Packet: {dns_query.hex()}")

if __name__ == "__main__":
    main()