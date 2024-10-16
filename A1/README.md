# Ecse316 DNS Client Program
DnsClient is a command-line application for querying DNS servers. It allows users to send DNS queries and receive responses for various record types, including A, MX, and NS records.

## Command Line Arguments
-  -t timeout (optional): Specifies how long to wait, in seconds, before retransmitting an unanswered query. Default value: 5.

-   -r max-retries (optional): Maximum number of times to retransmit an unanswered query before giving up. Default value: 3.

-   -p port (optional): UDP port number of the DNS server. Default value: 53.

-   -mx (optional): Sends a MX (mail server) query.

-   -ns (optional): Sends a NS (name server) query.

-   @server (required): The IPv4 address of the DNS server in a.b.c.d format.

-  name (required): The domain name to query for.

Running the client program:

```python dnsClient.py [-t timeout] [-r max-retries] [-p port] [-mx|-ns] @server name```

Examples of basic A record query

```python dnsClient.py @8.8.8.8 www.example.com```

## Mock DNS Server Program for Testing
Running the local dns server:

```python dsnServer.py```

This will start the DNS server on 127.0.0.1 at port 8080 and respond with the IP address 192.168.1.1 for valid queries.

The server can respond with the following DNS error codes based on the requested domain:

-  Format Error (1): Returned for the domain format.error.

-  Server Failure (2): Returned for the domain server.failure.

-  Not Implemented (4): Returned for the domain not.implemented.

-  Refused (5): Returned for the domain refused.query.


Querying from the local dns server with refused error: 

```python dnsClient.py  -p 8080 "@127.0.0.1" refused.query```

Python version used: 3.9.6+
