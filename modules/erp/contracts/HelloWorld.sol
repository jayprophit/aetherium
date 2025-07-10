// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract HelloWorld {
    string public message = "Hello, ERP Blockchain!";
    
    function setMessage(string calldata newMessage) external {
        message = newMessage;
    }
}
