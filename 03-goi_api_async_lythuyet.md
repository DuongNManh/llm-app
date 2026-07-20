# goi api bat dong bo

## httpx

- la thu vien http client hien dai cua python
- ho tro ca synchronous va asynchoronous
- chiu trach nhiem: mo ket noi, gui request, nhan response, parse JSON, quan ly connection
=> ta chi can goi await client.get()

### Client

- doi tuong dai dien cho ket noi cua app ra dich vu ben ngoai
- App -> AsyncClient -> Internet -> API Server
- No quan ly: DNS cache, TCP connection, SSL, Cookie, Header, Connection Pool
- Khong nen tao client trong vong lap (loop 100 lan tao 100 connection -> lang phi tai nguyen)
- trong thuc te: 1 AsyncClient -> (N request) -> dong Client

### Connection pool

- do AsyncClient quan ly
- reuse connection dang free trong pool
