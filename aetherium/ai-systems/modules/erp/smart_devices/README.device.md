# ERP Smart Device Integration Example (Integrated)

This folder contains code samples and SDK integrations for smart devices (e.g., IoT, smartwatches, smart TVs, etc.) as part of the knowledge-base ERP module.

## Example: Node.js Device Stub
```js
// device_stub.js
const axios = require('axios');

async function sendHeartbeat() {
  await axios.post('https://api.knowledge-base/erp/heartbeat', { deviceId: 'device-001', status: 'online' });
}

setInterval(sendHeartbeat, 60000);
```

## Example: Embedded C (Zephyr RTOS)
```c
#include <zephyr/device.h>
#include <zephyr/kernel.h>

void main(void) {
  while (1) {
    // Send heartbeat to ERP backend
    k_sleep(K_SECONDS(60));
  }
}
```
