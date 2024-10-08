import struct
import socket
import time

def parse_dns_response(response, query_name, server_ip, request_type, start_time, retries):
    def read_name(response, offset):
        labels = []
        while True:
            length = response[offset]
            if length & 0xC0 == 0xC0:  # Pointer
                pointer = struct.unpack(">H", response[offset:offset + 2])[0]
                offset += 2
                return read_name(response, pointer & 0x3FFF)[0], offset
            if length == 0:  # End of name
                offset += 1
                break
            offset += 1
            labels.append(response[offset:offset + length].decode())
            offset += length
        return ".".join(labels), offset

    try:
        # Parse the DNS header
        header = response[:12]
        (
            transaction_id,  # ID
            flags,           # Flags
            questions,       # Number of Questions
            answer_rrs,      # Number of Answer Resource Records
            authority_rrs,   # Number of Authority Resource Records
            additional_rrs   # Number of Additional Resource Records
        ) = struct.unpack(">HHHHHH", header)
        
        # Print query summary
        print(f"DnsClient sending request for {query_name}")
        print(f"Server: {server_ip}")
        print(f"Request type: {request_type}")

        # Calculate response time
        response_time = time.time() - start_time
        print(f"Response received after {response_time:.5f} seconds ({retries} retries)")

        # Skip the Question section
        index = 12
        for _ in range(questions):
            _, index = read_name(response, index)
            index += 4  # Skip QTYPE and QCLASS

        # Parse the Answer section
        if answer_rrs > 0:
            print(f"***Answer Section ({answer_rrs} records)***")
            for i in range(answer_rrs):
                # Parse the answer name
                name, index = read_name(response, index)
                
                # Parse the rest of the answer fields
                answer_type, answer_class, ttl, data_length = struct.unpack(">HHIH", response[index:index + 10])
                index += 10

                # Determine if the response is authoritative
                auth = "auth" if (flags & 0x0400) else "nonauth"

                # If it's an A record (IPv4 address)
                if answer_type == 1:  # Type A
                    ip_address = socket.inet_ntoa(response[index:index + data_length])
                    print(f"IP\t{ip_address}\t{ttl}\t{auth}")
                elif answer_type == 5:  # Type CNAME
                    cname, _ = read_name(response, index)
                    print(f"CNAME\t{cname}\t{ttl}\t{auth}")
                elif answer_type == 15:  # Type MX
                    preference = struct.unpack(">H", response[index:index + 2])[0]
                    mx, _ = read_name(response, index + 2)
                    print(f"MX\t{mx}\t{preference}\t{ttl}\t{auth}")
                elif answer_type == 2:  # Type NS
                    ns, _ = read_name(response, index)
                    print(f"NS\t{ns}\t{ttl}\t{auth}")
                else:
                    print("  Non-A record type")
                
                index += data_length
        else:
            print("NOTFOUND")

        # Parse the Additional section
        if additional_rrs > 0:
            print(f"***Additional Section ({additional_rrs} records)***")
            for i in range(additional_rrs):
                # Parse the additional record name
                name, index = read_name(response, index)
                
                # Parse the rest of the additional record fields
                additional_type, additional_class, ttl, data_length = struct.unpack(">HHIH", response[index:index + 10])
                index += 10

                # Determine if the response is authoritative
                auth = "auth" if (flags & 0x0400) else "nonauth"

                # If it's an A record (IPv4 address)
                if additional_type == 1:  # Type A
                    ip_address = socket.inet_ntoa(response[index:index + data_length])
                    print(f"IP\t{ip_address}\t{ttl}\t{auth}")
                elif additional_type == 5:  # Type CNAME
                    cname, _ = read_name(response, index)
                    print(f"CNAME\t{cname}\t{ttl}\t{auth}")
                elif additional_type == 15:  # Type MX
                    preference = struct.unpack(">H", response[index:index + 2])[0]
                    mx, _ = read_name(response, index + 2)
                    print(f"MX\t{mx}\t{preference}\t{ttl}\t{auth}")
                elif additional_type == 2:  # Type NS
                    ns, _ = read_name(response, index)
                    print(f"NS\t{ns}\t{ttl}\t{auth}")
                else:
                    print("  Non-A record type")
                
                index += data_length
    except Exception as e:
        print(f"ERROR\t{str(e)}")