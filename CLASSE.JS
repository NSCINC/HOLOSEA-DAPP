class ClassNamespace {
    constructor(name) {
        this.name = name;
        this.proxies = new Map();
    }

    getName() {
        return this.name;  // Corrigido o método para retornar o nome
    }

    getProxies() {
        return this.proxies;
    }

    addProxy(className, address) {
        this.proxies.set(className, address); // Método para adicionar proxy
    }

    getProxy(className) {
        return this.proxies.get(className); // Método para obter proxy específico
    }
}

class ClassNamespaceManager {
    constructor() {
        this.namespaces = new Map();
    }

    // Evento emitido ao carregar uma nova biblioteca
    libraryLoaded(path) {
        console.log(`LibraryLoaded event emitted: ${path}`);
    }

    // Função para obter um proxy (endereço) de uma classe em um namespace
    getClassProxy(namespace, className) {
        const ns = this.namespaces.get(namespace);
        return ns ? ns.getProxy(className) : null; // Usando o novo método getProxy
    }

    // Função para carregar uma biblioteca compartilhada
    loadLibrary(path) {
        this.libraryLoaded(path); // Emitir evento de biblioteca carregada
        // Aqui você pode adicionar a lógica para carregar a biblioteca nativa se necessário
    }

    // Função para criar um novo namespace de classes
    createClassNamespace(namespaceName) {
        const newNamespace = new ClassNamespace(namespaceName);
        this.namespaces.set(namespaceName, newNamespace);
        console.log(`Namespace '${namespaceName}' criado`);
    }
}

// Exemplo de uso das funções
const manager = new ClassNamespaceManager();

// Criar um namespace
manager.createClassNamespace("NamespaceA");

// Adicionar um proxy para uma classe no NamespaceA
const ns = manager.namespaces.get("NamespaceA");
if (ns) {
    ns.addProxy("ClassNameA", "0xAddress"); // Usando o novo método addProxy
}

// Obter o proxy para a classe ClassNameA no NamespaceA
const proxy = manager.getClassProxy("NamespaceA", "ClassNameA");
console.log(`Proxy para 'ClassNameA' no 'NamespaceA': ${proxy}`);

// Carregar uma biblioteca compartilhada
manager.loadLibrary("path/to/library.so");
